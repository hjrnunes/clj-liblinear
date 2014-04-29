(ns clj-liblinear.core
  (:refer-clojure :exclude [read-string])
  (:require [clojure.set :refer [union]]
            [clojure.edn :refer [read-string]]
            clojure.core.matrix
            clojure.core.matrix.impl.dataset)
  (:import (de.bwaldvogel.liblinear FeatureNode
                                    Model
                                    Linear
                                    Problem
                                    Parameter
                                    SolverType)))

(set! *warn-on-reflection* true)

(defprotocol FeatureRows
  (construct-feature-nodes-arrays [this] "Construct an array of arrays of FeatureNode objects (representing rows).")
  (get-dimensions [this] "Get all of the dimensions as a map of dimension -> index."))

(extend-protocol FeatureRows
  ;; a sequence of map/sets instances
  clojure.lang.ISeq
  (construct-feature-nodes-arrays [iseq] nil)
  (get-dimensions [iseq]
    (let [dimnames (cond (every? map? iseq) (into #{} (flatten (map keys iseq)))
                         (every? set? iseq) (apply union iseq))]
      (into {} (map vector
                    dimnames
                    (range 1
                           (inc (count dimnames)))))))
  ;; a core.matrix Dataset
  clojure.core.matrix.impl.dataset.DataSet
  (construct-feature-nodes-arrays [this] nil)
  (get-dimensions [this] nil))

(defprotocol FeatureRow
  (construct-feature-nodes [this dimensions]
    "Represent this as a sequence of FeatureNode objects."))

(extend-protocol FeatureRow
  ;; a map
  clojure.lang.IPersistentMap
  (construct-feature-nodes [this dimensions]
    (for [[k v] this
          :when (contains? dimensions k)]
      (FeatureNode. (get dimensions k) v)))
  ;; a set
  clojure.lang.IPersistentSet
  (construct-feature-nodes [this dimensions]
    (for [v this
          :when (dimensions v)]
      (FeatureNode. (get dimensions v) 1))))

(defn- construct-bias-feature [bias feature-index] (FeatureNode. feature-index bias))

(defn construct-feature-nodes-array
  "Given a FeatureRow, represent it as an array of FeatureNode objects,
ordered by index (the order of dimensions) and including the bias if necessary.
If bias is active, an extra feature is added."
  [bias dimensions feature-row]
  (let [ordered-nodes (sort-by #(.index ^FeatureNode %)
                               (construct-feature-nodes feature-row
                                                        dimensions))]
    (if (>= bias 0)
      (into-array (concat ordered-nodes
                          [(construct-bias-feature bias
                                                   (inc (count dimensions)))]))
      (into-array ordered-nodes))) )


(defn- count-correct-predictions
  [target labels]
  (count (filter true? (map = target labels))))

(defn train
  "Train a LIBLINEAR model on a collection of maps or sets, xs, and a collection of their integer classes, ys."
  [xs ys & {:keys [c eps algorithm bias cross-fold]
            :or {c 1, eps 0.1, algorithm :l2l2, bias -1, cross-fold nil}}]
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
        bias 	   (if (= Boolean (type bias))
                     (if (true? bias) 1 -1)
                     (if (>= bias 0) bias -1))
        dimensions (get-dimensions xs)
        feature-nodes-arrays (construct-feature-nodes-arrays xs)  
        ;;(into-array (map #(feature-array bias dimensions %) xs))
        ys         (into-array Double/TYPE ys)
        prob       (new Problem)]

    (set! (.x prob) feature-nodes-arrays)
    (set! (.y prob) ys)
    (set! (.bias prob) bias)
    (set! (.l prob) (count feature-nodes-arrays))
    (set! (.n prob) (+ (count dimensions) (if (>= bias 0) 1 0)))
    
    ;;Train and return the model
    {:target          (when cross-fold 
                        (let [target (make-array Double/TYPE (count ys))]
                          (Linear/crossValidation prob params cross-fold target)
                          (println (format "Cross Validation Accuracy = %g%%%n"
                                           (* 100.0 (/ (count-correct-predictions target ys) (count target)))))
                          target))
     :liblinear-model (Linear/train prob params)
     :dimensions dimensions}))

(defn predict [model feature-row]
  (let [m ^Model (:liblinear-model model)]
    (Linear/predict m (construct-feature-nodes-array (.getBias m)
                                                     (:dimensions model)
                                                     feature-row))))

(defn save-model
  "Writes the model out to two files specified by the base-file-name which should be a path and base file name. The extention .bin is added to the serialized java model and .edn is added to the clojure dimensions data."
  [model base-file-name]
  (with-open [out-file (clojure.java.io/writer (str base-file-name ".bin"))]
    (Linear/saveModel out-file ^Model (:liblinear-model model)))
  (spit  (str base-file-name ".edn") (:dimensions model)))

(defn load-model
  "Reads a useable model from a pair of files specified by base-file-name. A file with the .bin extension should contain the serialized java model and the .edn file should contain the serialized (edn) clojure dimensions data."
  [base-file-name]
  (let [mdl (Linear/loadModel (clojure.java.io/reader (str base-file-name ".bin")))
        dimensions (read-string (slurp (str base-file-name ".edn")))]
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
        feature-indices (if include-bias
                          (assoc (:dimensions model)
                            ;; The bias feature is always the last one.
                            :bias (count coefficients-vector))
                          (:dimensions model))
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
