#include <catch2/catch.hpp>

#include <core/dataset.h>
#include <core/classifier.h>
#include <core/classifier.h>
#include <string>
#include <sstream>
#include <limits>

/**
 * Run Command:
 * ./Debug/naive-bayes-test/naive-bayes-test.app/Contents/MacOS/naive-bayes-test
 * under the directory: naivebayes-mingw4/cmake-build-default
 */


TEST_CASE("Check classifier valid") {

  // load training data and testing data
  naivebayes::ImageDataset dataset("../data/mnistdatatraining/", true);
  naivebayes::ImageDataset test_dataset("../data/testimagesandlabels.txt", false);
  naivebayes::Classifier classifier;

  // using classifier to fit the training data
  classifier.Fit(dataset);

  // Caculate the accuracy on test dataset
  float acc = classifier.Validate(test_dataset);

  // assert the accuracy is greater than 0.7
  REQUIRE(acc >= 0.7f);

}
