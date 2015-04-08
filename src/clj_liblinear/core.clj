(ns clj-liblinear.core
  (:refer-clojure :exclude [read-string])
  (:require [clojure.set :refer [union]]
            [clojure.edn :refer [read-string]]
            [clojure.core.matrix :as m]
            [clojure.core.matrix.impl.dataset :as d]
            [clatrix.core :as clx])
  (:import (de.bwaldvogel.liblinear FeatureNode
                                    Model
                                    Linear
                                    Problem
                                    Parameter
                                    SolverType)
           (clojure.lang IPersistentVector
                         IPersistentMap)))

(set! *warn-on-reflection* true)


(defrecord IndexedValues
  ;; "a sequence of values, together with an index -- a map from each value to its ordinal number"
  [^IPersistentVector values
   ^IPersistentMap index])

(defn indexed-values
  ([values index]
   (IndexedValues. values
                   index))
  ([values]
   (indexed-values values
                   (into {} (map vector
                                 values
                                 (range 1
                                        (inc (count values))))))))

(defn add [^IndexedValues ivs
           new-val]
  (IndexedValues. (conj (:values ivs)
                        new-val)
                  (assoc (:index ivs)
                    new-val (inc (count (:values ivs))))))


(defprotocol FeatureRows
  (construct-feature-nodes-arrays [this bias dimensions] "Construct an array of arrays of FeatureNode objects (representing rows).")
  (get-dimensions [this] "Get all of the dimensions, indexed."))

(defprotocol FeatureRow
  (construct-feature-nodes [this dimensions]
                           "Represent this as a sequence of FeatureNode objects."))

(defn- construct-bias-feature [bias feature-index] (FeatureNode. feature-index bias))

(defn construct-feature-nodes-array
  "Given a FeatureRow, represent it as an array of FeatureNode objects,
ordered by index (the order of dimensions) and including the bias if necessary.
If bias is active, an extra feature is added."
  [feature-row bias dimensions]
  (let [ordered-nodes (sort-by #(.index ^FeatureNode %)
                               (construct-feature-nodes feature-row
                                                        dimensions))]
    (if (>= bias 0)
      (into-array (concat ordered-nodes
                          [(construct-bias-feature bias
                                                   (inc (count (:values dimensions))))]))
      (into-array ordered-nodes))))

(extend-protocol FeatureRows
  ;; a sequence of map/sets instances
  clojure.lang.ISeq
  (construct-feature-nodes-arrays [iseq bias dimensions]
    (into-array (map #(construct-feature-nodes-array %
                                                     bias
                                                     dimensions)
                     iseq)))
  (get-dimensions [iseq]
    (let [dimnames (cond (every? map? iseq) (into #{} (flatten (map keys iseq)))
                         (every? set? iseq) (apply union iseq))]
      (indexed-values dimnames)))
  ;; a core.matrix Dataset
  clojure.core.matrix.impl.dataset.DataSet
  (construct-feature-nodes-arrays [dat bias dimensions]
    (let [transposed-mat (clx/matrix (:columns dat))]
      (into-array (map #(construct-feature-nodes-array (double-array %)
                                                       bias
                                                       dimensions)
                       (m/columns transposed-mat)))))
  (get-dimensions [dat]
    (indexed-values (set (:column-names dat)))))

(extend-protocol FeatureRow
  ;; a map
  clojure.lang.IPersistentMap
  (construct-feature-nodes [this dimensions]
    (for [[k v] this
          :let [k-idx (get (:index dimensions) k)]
          :when k-idx]
      (FeatureNode. k-idx
                    v)))
  ;; a set
  clojure.lang.IPersistentSet
  (construct-feature-nodes [this dimensions]
    (for [v this
          :let [v-idx (get (:index dimensions) v)]
          :when v-idx]
      (FeatureNode. v-idx
                    1))))

(extend-protocol FeatureRow
  ;; a primitive array of doubles
  ;; http://stackoverflow.com/a/13925248
  (Class/forName "[D")
  (construct-feature-nodes [this dimensions]
    (for [idx (range (count this))]
      (FeatureNode. (inc idx)
                    (aget ^doubles this
                          ^int idx)))))

(defn- count-correct-predictions
  [target labels]
  (count (filter true? (map = target labels))))

(defn train
  "Train a LIBLINEAR model on a collection of maps or sets, xs, and a collection of their integer classes, ys."
  [xs ys & {:keys [c eps algorithm bias cross-fold keep-nan-ys]
            :or   {c 1, eps 0.1, algorithm :l2l2, bias -1, cross-fold nil, keep-nan-ys false}}]
  (let [params (new Parameter (condp = algorithm
                                :l2lr_primal SolverType/L2R_LR
                                :l2l2 SolverType/L2R_L2LOSS_SVC_DUAL
                                :l2l2_primal SolverType/L2R_L2LOSS_SVC
                                :l2l1 SolverType/L2R_L1LOSS_SVC_DUAL
                                :multi SolverType/MCSVM_CS
                                :l1l2_primal SolverType/L1R_L2LOSS_SVC
                                :l1lr SolverType/L1R_LR
                                :l2lr SolverType/L2R_LR)
                    c eps)
        bias (if (= Boolean (type bias))
               (if (true? bias) 1 -1)
               (if (>= bias 0) bias -1))
        dimensions (get-dimensions xs)
        xs (construct-feature-nodes-arrays xs
                                           bias
                                           dimensions)
        ys (double-array ys)
        ;; handling the case of nan ys (interpreted as multiple
        ;; classes if kept)
        [xs ys] (if keep-nan-ys
                  [xs ys]
                  ;; else -- remove cases where y is nan
                  (->> (map vector xs ys)
                       (filter #(not (Double/isNaN (second %))))
                       ((fn [xys]
                          [(into-array (map first xys))
                           (double-array (map second xys))]))))
        prob (new Problem)]

    (set! (.x prob) xs)
    (set! (.y prob) ys)
    (set! (.bias prob) bias)
    (set! (.l prob) (count xs))
    (set! (.n prob) (+ (count (:index dimensions)) (if (>= bias 0) 1 0)))

    ;;Train and return the model
    {:target          (when cross-fold
                        (let [target (make-array Double/TYPE (count ys))]
                          (Linear/crossValidation prob params cross-fold target)
                          (println (format "Cross Validation Accuracy = %g%%%n"
                                           (* 100.0 (/ (count-correct-predictions target ys) (count target)))))
                          target))
     :liblinear-model (Linear/train prob params)
     :dimensions      dimensions}))

(defn predict [model feature-row]
  (let [m ^Model (:liblinear-model model)]
    (Linear/predict m (construct-feature-nodes-array feature-row
                                                     (.getBias m)
                                                     (:dimensions model)))))

(defn save-model
  "Writes the model out to two files specified by the base-file-name which should be a path and base file name. The extention .bin is added to the serialized java model and .edn is added to the clojure dimensions data."
  [model base-file-name]
  (with-open [out-file (clojure.java.io/writer (str base-file-name ".bin"))]
    (Linear/saveModel out-file ^Model (:liblinear-model model)))
  (let [dimensions (:dimensions model)]
    (spit (str base-file-name ".edn")
          {:values (:values dimensions)
           :index  (:index dimensions)})))

(defn load-model
  "Reads a useable model from a pair of files specified by base-file-name. A file with the .bin extension should contain the serialized java model and the .edn file should contain the serialized (edn) clojure dimensions data."
  [base-file-name]
  (let [mdl (Linear/loadModel (clojure.java.io/reader (str base-file-name ".bin")))
        dimensions-map (read-string (slurp (str base-file-name ".edn")))
        dimensions (indexed-values (:values dimensions-map)
                                   (:index dimensions-map))]
    {:liblinear-model mdl :dimensions dimensions}))

(defn get-coefficients
  "Get the nonzero coefficients of a given model, represented as a map from feature name to coefficient value.
The intercept corresponds to the key :intercept."
  [model]
  (let [bias (.getBias ^de.bwaldvogel.liblinear.Model (:liblinear-model model))
        ;; Check if the model contains a bias coefficient (intercept).
        include-bias (<= 0 bias)
        ;; Get a vector of the coefficients (ordered as in the
        ;; internal liblinear representation.
        coefficients-vector (-> model
                                :liblinear-model
                                (#(.getFeatureWeights
                                   ^de.bwaldvogel.liblinear.Model %))
                                vec)
        ;; Get the indices (in the above ordering) corresponding to
        ;; the various feature names.
        feature-indices (:index
                          (if include-bias
                            (add (:dimensions model)
                                 :bias)
                            (:dimensions model)))
        ;; Create a hashmap containing the coefficients by name.
        feature-coefficients (into {}
                                   (for [[feature-name feature-index] feature-indices
                                         :let [coefficient (coefficients-vector
                                                             ;; dec, to start from 0, not 1
                                                             (dec feature-index))]
                                         :when (not (zero? coefficient))]
                                     [feature-name coefficient]))]
    ;; Return the coefficients.
    (if-let [bias-coefficient (:bias feature-coefficients)]
      ;; If there is a bias coefficient, replace it with the intercept
      ;; (defined as the bias coefficient multiplied by the constant
      ;; value of the bias feature).
      (assoc (dissoc feature-coefficients
                     :bias)
        :intercept (* bias-coefficient bias))
      ;; Otherwise, just return.
      feature-coefficients)))


(defn reset-random
  "Reset the PRNG used by liblinear.
This is useful for regression tests and for reproducibility of experiments."
  []
  (Linear/resetRandom))
