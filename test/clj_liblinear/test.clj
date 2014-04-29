(ns clj-liblinear.test
  (:use [clj-liblinear.core :only [train predict get-coefficients reset-random]]
        clojure.test)
  (:import java.util.Random))


(comment
  (let [train-data (concat
                    (repeatedly 300 #(hash-map :class 0 :f {:x (rand), :y (rand)}))
                    (repeatedly 300 #(hash-map :class 1 :f {:x (- (rand)), :y (- (rand))})))
        model (train
               (map :f train-data)
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
                    {:f features
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

(defn regression-test-template [& regression-cases]
  ;; Check the model coefficients for various training scenations:
  (eval (concat `(clojure.test/are [training-parameters expected-coefficients]
                   (almost-equal-maps (do
                                        ;; Reset liblinear's PRNG
                                        (clj-liblinear.core/reset-random)
                                        ;; Train model and get coefficients
                                        (clj-liblinear.core/get-coefficients (apply clj-liblinear.core/train
                                                                                    (map :f train-data)
                                                                                    (map :class train-data)
                                                                                    training-parameters)))
                                      expected-coefficients))
                regression-cases)))


(deftest regression-test-for-git-2266be6
  ;; Test various combinations of algorithm (taken from the supported
  ;; algotithms), c (taken from #{2, 1/2}) and bias (taken from
  ;; #{1, true, -1}.
  (regression-test-template [:algorithm :l2lr_primal
                             :c 2
                             :bias true]
                            {:intercept 0.8451298717761591,
                             :y 0.9234249966851159,
                             :x -2.0443777978405175}
                            [:algorithm :l2l2
                             :c 1/2
                             :bias 1]
                            {:intercept 0.30896403558307006,
                             :y 0.32897934116468414,
                             :x -0.7493852105195001}
                            [:algorithm :l2l2
                             :c 1/2
                             :bias -1]
                            {:y 0.3071169929022199,
                             :x -0.6748243079870543}
                            [:algorithm :l2l2_primal
                             :c 2
                             :bias 1]
                            {:intercept 0.29480508710912584,
                             :y 0.3137835397140601,
                             :x -0.7111387065073224}
                            [:algorithm :l2l2_primal
                             :c 2
                             :bias -1]
                            {:y 0.30033425626017296,
                             :x -0.662693956280086}
                            [:algorithm :l2l1
                             :c 1/2
                             :bias true]
                            {:intercept 0.620462367740796,
                             :y 0.734783428258156,
                             :x -1.6304922916122815}
                            [:algorithm :multi
                             :c 2
                             :bias true]
                            {:intercept -0.34464667291199613,
                             :y 0.3979628683341206,
                             :x -0.3979628683341171}
                            [:algorithm :l1l2_primal
                             :c 1/2
                             :bias 1]
                            {:intercept 0.2886149834553072,
                             :y 0.308149642263233,
                             :x -0.7236401689670875}
                            [:algorithm :l1l2_primal
                             :c 1/2
                             :bias -1]
                            {:y 0.300478785493661,
                             :x -0.6696216144029736}
                            [:algorithm :l1lr
                             :c 2
                             :bias 1]
                            {:intercept 0.7529187765874954,
                             :y 0.8761760796248441,
                             :x -1.9341912291944392}
                            [:algorithm :l1lr
                             :c 2
                             :bias -1]
                            {:y 0.8430193551171671,
                             :x -1.8029098028343824}
                            [:algorithm :l2lr
                             :c 1/2
                             :bias true]
                            {:intercept 0.7905463338806276,
                             :y 0.8582570400126215,
                             :x -1.917358197930745}))

