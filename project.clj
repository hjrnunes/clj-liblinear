(defproject hjrnunes/clj-liblinear "0.1.5"
            :description "A Clojure wrapper for LIBLINEAR, a linear support vector machine library."
            :url "https://github.com/hjrnunes/clj-liblinear"
            :license {:name "Eclipse Public License - v 1.0"
                      :url  "https://www.eclipse.org/legal/epl-v10.html"}
            :scm {:name "git"
                  :url  "https://github.com/hjrnunes/clj-liblinear"}
            :signing {:gpg-key "8896C73B"}
            :deploy-repositories [["clojars" {:creds :gpg}]]

            :dependencies [[org.clojure/clojure "1.6.0"]
                           [de.bwaldvogel/liblinear "1.94"]
                           [net.mikera/core.matrix "0.23.0"]
                           [clatrix "0.3.0"]])
