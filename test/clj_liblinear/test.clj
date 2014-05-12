(ns clj-liblinear.test
  (:use clj-liblinear.core
        clojure.test)
  (:require [clojure.core.matrix.impl.dataset :as d])
  (:import java.util.Random))


(comment
  (let [train-data (concat
                    (repeatedly 300 #(hash-map :class 0 :features {:x (rand), :y (rand)}))
                    (repeatedly 300 #(hash-map :class 1 :features {:x (- (rand)), :y (- (rand))})))
        model (train
               (map :features train-data)
               (map :class train-data)
               :algorithm :l2l2)]
           
    [(predict model {:x (rand) :y (rand)})
     (predict model {:x (- (rand)) :y (- (rand))})]))
;;=> [0 1]


(defn inverse-logit [x]
  (/ (inc (Math/exp (- x)))))


(defn generate-logistic-observations
  "Generate n pseudorandom observations by a logistic model.
The feature values are iid samples of standard normal distribution.
The model coefficients are specified as a map from feature name to coefficient value.
The intercept is specified in feature name :intercept."
  [n coefficients seed]
  (let [;; To make the test consistent, we do not use clojure.core's
        ;; usual rand function here.
        ;; One might consider using org.clojure/data.generators for such
        ;; needs in the future.
        prng (Random. seed)
        rand-normal #(.nextGaussian ^Random prng)
        rand-uniform #(.nextDouble ^Random prng)
        feature-names (keys (dissoc coefficients
                                    :intercept))
        intercept (or (:intercept coefficients)
                      0)]
    (repeatedly n
                (fn []
                  (let [features (into {}
                                       (for [feature-name feature-names]
                                         [feature-name (rand-normal)]))
                        prob (inverse-logit
                              (reduce +
                                      (cons intercept
                                            (for [[feature-name feature-value] features]
                                              (* (coefficients feature-name)
                                                 feature-value)))))
                        observed-class (Math/signum (- prob
                                                       (rand-uniform)))]
                    {:features features
                     :class observed-class})))))


(defn almost-equal-numbers
  "Given some numbers, check if they are equal up to small relative error."
  [& xs]
  (or (every? zero? xs)
      (let [x0 (first xs)]
        (and (not (zero? x0)))
        (let [allowed-error (* (Math/abs x0)
                               0.000001)]
          (every? (fn [x] (let [err (- x x0)]
                           (< (- allowed-error)
                              err
                              allowed-error)))
                  (rest xs))))))

(deftest almost-equal-numbers-test
  (is (almost-equal-numbers 1
                            (+ 1 0.0000001)))
  (is (not (almost-equal-numbers 0
                                 0.0000001))))

(defn almost-equal-maps
  "Given a sequence of key-value maps whose values are numbers, check if they are all equal up to small relative erros of their values (relative wrt the values of the first map)."
  [& ms]
  (and (apply = (map #(apply hash-set (keys %))
                     ms))
       (every? (fn [k]
                 (apply almost-equal-numbers (map #(% k) ms)))
               (keys (first ms)))))

(deftest almost-equal-maps-test
  (is (almost-equal-maps {"abc" 1}
                         {"abc" (+ 1 0.0000001)}
                         {"abc" (- 1 0.0000001)}))
  (is (almost-equal-maps {"abc" -1}
                         {"abc" (+ -1 0.0000001)}
                         {"abc" (- -1 0.0000001)}))
  (is (not (almost-equal-maps {"abc" 1}
                              {"abc" (+ 1 0.0000001)}
                              {"abc" 1 "def" 1})))
  (is (not (almost-equal-maps {"abc" 1}
                              {"abc" (+ 1 0.0001)})))
  (is (almost-equal-maps {:intercept 0.7529187765874954, :y 0.8761760796248441, :x -1.9341912291944392}
                         {:intercept 0.7529187765874954, :y 0.8761760796248441, :x -1.9341912291944392})))

(def train-data (generate-logistic-observations 400
                                                {:x -2
                                                 :y 1
                                                 :intercept 1}
                                                0))

(def negated-train-data (map #(update-in % [:class] -)
                             train-data))


(defn regression-test-template [& test-cases]
  ;; Check the model coefficients for various training scenations:
  (eval (concat `(clojure.test/are [training-parameters expected-coefficients]
                   (almost-equal-maps (do
                                        ;; Reset liblinear's PRNG
                                        (clj-liblinear.core/reset-random)
                                        ;; Train model and get coefficients
                                        (clj-liblinear.core/get-coefficients (apply clj-liblinear.core/train
                                                                                    (map :features train-data)
                                                                                    (map :class train-data)
                                                                                    training-parameters)))
                                      expected-coefficients))
                (apply concat test-cases))))


(def git-2266be6-test-cases
  ;; Test various combinations of algorithm (taken from the supported
  ;; algotithms), c (taken from #{2, 1/2}) and bias (taken from
  ;; #{1, true, -1}.
  [[[:algorithm :l2lr_primal
     :c 2
     :bias true]
    {:intercept 0.8451298717761591, :y 0.9234249966851159, :x -2.0443777978405175}]
   [[:algorithm :l2l2
     :c 1/2
     :bias 1]
    {:intercept 0.30896403558307006, :y 0.32897934116468414, :x -0.7493852105195001}]
   [[:algorithm :l2l2
     :c 1/2
     :bias -1]
    {:y 0.3071169929022199, :x -0.6748243079870543}]
   [[:algorithm :l2l2_primal
     :c 2
     :bias 1]
    {:intercept 0.29480508710912584, :y 0.3137835397140601, :x -0.7111387065073224}]
   [[:algorithm :l2l2_primal
     :c 2
     :bias -1]
    {:y 0.30033425626017296, :x -0.662693956280086}]
   [[:algorithm :l2l1
     :c 1/2
     :bias true]
    {:intercept 0.620462367740796, :y 0.734783428258156, :x -1.6304922916122815}]
   [[:algorithm :multi
     :c 2
     :bias true]
    {:intercept -0.34464667291199613, :y 0.3979628683341206, :x -0.3979628683341171}]
   [[:algorithm :l1l2_primal
     :c 1/2
     :bias 1]
    {:intercept 0.2886149834553072, :y 0.308149642263233, :x -0.7236401689670875}]
   [[:algorithm :l1l2_primal
     :c 1/2
     :bias -1]
    {:y 0.300478785493661, :x -0.6696216144029736}]
   [[:algorithm :l1lr
     :c 2
     :bias 1]
    {:intercept 0.7529187765874954, :y 0.8761760796248441, :x -1.9341912291944392}]
   [[:algorithm :l1lr
     :c 2
     :bias -1]
    {:y 0.8430193551171671, :x -1.8029098028343824}]
   [[:algorithm :l2lr
     :c 1/2
     :bias true]
    {:intercept 0.7905463338806276, :y 0.8582570400126215, :x -1.917358197930745}]])


(def git-e5ac6ff-test-cases
  ;; Test various combinations of algorithm (taken from the supported
  ;; algotithms), c (taken from #{2, 1/2}) and bias (taken from
  ;; #{false, 0, 1/2, 2}.
  [[[:algorithm :l2lr_primal
      :c 2
     :bias false]
    {:y 0.8526765355183481, :x -1.8337994601561565}]
   [[:algorithm :l2l2
      :c 1/2
     :bias 0]
    {:y 0.3071169929022199, :x -0.6748243079870543}]
   [[:algorithm :l2l2
      :c 1/2
     :bias 1/2]
    {:intercept 0.3052391093981819, :y 0.3316509018715258, :x -0.7459067206733213}]
   [[:algorithm :l2l2_primal
      :c 2
     :bias 2]
    {:intercept 0.2982394215552014, :y 0.3212333172919317, :x -0.7092266053362317}]
   [[:algorithm :l2l2_primal
      :c 2
     :bias false]
    {:y 0.30033425626017296, :x -0.662693956280086}]
   [[:algorithm :l2l1
      :c 1/2
     :bias 0]
    {:y 0.6069312911692889, :x -1.3852881423300722}]
   [[:algorithm :multi
      :c 2
     :bias 1/2]
    {:intercept -0.32057433223338094, :y 0.4075424968639208, :x -0.40754249686392163}]
   [[:algorithm :l1l2_primal
      :c 1/2
     :bias 2]
    {:intercept 0.2902361764826205, :y 0.30826752895573417, :x -0.7241907729810667}]
   [[:algorithm :l1l2_primal
      :c 1/2
     :bias false]
    {:y 0.300478785493661, :x -0.6696216144029736}]
   [[:algorithm :l1lr
      :c 2
     :bias 0]
    {:y 0.8302090159480037, :x -1.7793320288789982}]
   [[:algorithm :l1lr
      :c 2
     :bias 1/2]
    {:intercept 0.7441257065436431, :y 0.8748933116220571, :x -1.9307406585250886}]
   [[:algorithm :l2lr
      :c 1/2
     :bias 2]
    {:intercept 0.8126707014260155, :y 0.8625038218399448, :x -1.9281362351975666}]])


(def all-regression-test-cases
  (concat git-2266be6-test-cases
          git-e5ac6ff-test-cases))


(deftest regression-test
  (apply regression-test-template
         all-regression-test-cases))



(comment
  ;; See what happens when labels are negated
  (clojure.pprint/pprint
   (let [reports
         (for [[training-parameters expected-coefficients] all-regression-test-cases]
           (let [expected-neg-coefficients (into {}
                                                 (for [[k v] expected-coefficients]
                                                   {k (- v)}))
                 _ (clj-liblinear.core/reset-random)
                 actual-neg-coefficients (clj-liblinear.core/get-coefficients (apply clj-liblinear.core/train
                                                                                     (map :features negated-train-data)
                                                                                     (map :class negated-train-data)
                                                                                     training-parameters))
                 _ (assert (= (keys expected-neg-coefficients)
                              (keys actual-neg-coefficients)))
                 report {:training-parameters training-parameters
                         :almost-equal (almost-equal-maps expected-neg-coefficients
                                                          actual-neg-coefficients)
                         :relative-differences (into {}
                                                     (for [k (keys expected-neg-coefficients)]
                                                       {k (/ (- (actual-neg-coefficients k)
                                                                (expected-neg-coefficients k))
                                                             (expected-neg-coefficients k))}))}]
             report)
           )]     
     {:reports reports
      :max-abs-relative-error (apply max
                                     (map #(Math/abs %)
                                          (flatten
                                           (map (comp vals :relative-differences)
                                                reports))))}))
  ;; It turns out that the liblinear library is not completely
  ;; symmetric (at least with some of the models):
  ;; When training a model with the same parameters and same seed, at
  ;; the same training data with negated labels, we get coefficients
  ;; which are only approximately the negations of the original
  ;; coefficients (absolute relative differences can reach 0.07680557501022828, but
  ;; are usually less than 0.01 or even much smaller).
  ;; The direction of change is not consistent (at least with the toy
  ;; data used in this check).
  )


(deftest indexed-values-test
  (is (= (add (indexed-values [:c :a :b])
              :x)
         (indexed-values [:c :a :b :x],
                         {:c 1, :a 2, :b 3 :x 4}))))


(deftest dataset-test
  (let [distinct-feature-names (distinct (map (comp keys :features)
                                              train-data))
        _ (assert (= 1 (count distinct-feature-names)))
        feature-names (first distinct-feature-names)
        dat (d/dataset feature-names
                       (for [feature-name feature-names]
                         (map (comp feature-name :features)
                              train-data)))]
    (for [training-parameters (map first all-regression-test-cases)]
      (is (apply almost-equal-maps
                 (for [xs [(map :features train-data)
                           dat]]
                   (get-coefficients (do (reset-random)
                                         (apply clj-liblinear.core/train
                                                xs
                                                (map :class train-data)
                                                training-parameters)))))))))


(comment
 ;; some performance tests
  (let [num-rows 100000
        num-columns 150
        large-train-data (generate-logistic-observations
                          num-rows
                          (apply hash-map
                                 (interleave (cons :intercept
                                                   (map (comp keyword str)
                                                        (range num-columns)))
                                             (repeatedly (inc num-columns)
                                                         rand)))
                          0)
        feature-names (first (map (comp keys :features)
                                 large-train-data))
        dat (d/dataset feature-names
                      (for [feature-name feature-names]
                        (map (comp feature-name :features)
                             large-train-data)))
        training-parameters [:algorithm :l1lr :c 2 :bias 1/2]
        ys (map :class large-train-data)]
    (apply almost-equal-maps
           (for [xs [(map :features large-train-data)
                     dat]]
             (time (println (get-coefficients (do (reset-random)
                                                  (apply clj-liblinear.core/train
                                                         xs
                                                         ys
                                                         training-parameters)))))))))

