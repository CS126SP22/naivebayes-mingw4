#include <catch2/catch.hpp>

#include <core/dataset.h>
#include <core/classifier.h>
#include <core/classifier.h>
#include <string>
#include <sstream>

TEST_CASE("Check classifier computing") {

  naivebayes::ImageDataset dataset;
  naivebayes::Classifier classifier;

  std::string test_images_set = "#  \n#  \n#  \n###\n #\n###\n###\n  #\n  #\n";

  std::string test_labels_set = "1\n2\n7\n";

  std::istringstream ss(test_images_set);
  ss >> dataset;

  ss.clear();
  ss.str(test_labels_set);
  ss >> dataset;


  classifier.Fit(dataset);

  REQUIRE(classifier.Predict(dataset.GetImageAtIndex(0)) == 1);
  REQUIRE(classifier.Predict(dataset.GetImageAtIndex(1)) == 2);
  REQUIRE(classifier.Predict(dataset.GetImageAtIndex(2)) == 7);

}
