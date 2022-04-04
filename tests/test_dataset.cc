#include <catch2/catch.hpp>

#include <core/dataset.h>
#include <string>
#include <sstream>

TEST_CASE("Check dataset loading") {

  naivebayes::ImageDataset dataset;

  std::string test_images_set = "#  \n#  \n#  \n###\n #\n###\n###\n  #\n  #\n";

  std::string test_labels_set = "1\n2\n7\n";

  std::istringstream ss(test_images_set);
  ss >> dataset;

  ss.clear();
  ss.str(test_labels_set);
  ss >> dataset;

  REQUIRE(dataset.GetDatasetSize() == 3);
  REQUIRE(dataset.GetImageSize() == 3);
  naivebayes::image_t image = dataset.GetImageAtIndex(0);
  REQUIRE(image[0][0] == 1);
  REQUIRE(image[0][1] == 0);
  REQUIRE(image[0][2] == 0);
  REQUIRE(image[1][0] == 1);
  REQUIRE(image[1][1] == 0);
  REQUIRE(image[1][2] == 0);
  REQUIRE(image[2][0] == 1);
  REQUIRE(image[2][1] == 0);
  REQUIRE(image[2][2] == 0);
  image = dataset.GetImageAtIndex(1);
  REQUIRE(image[0][0] == 1);
  REQUIRE(image[0][1] == 1);
  REQUIRE(image[0][2] == 1);
  REQUIRE(image[1][0] == 0);
  REQUIRE(image[1][1] == 1);
  REQUIRE(image[1][2] == 0);
  REQUIRE(image[2][0] == 1);
  REQUIRE(image[2][1] == 1);
  REQUIRE(image[2][2] == 1);
  image = dataset.GetImageAtIndex(2);
  REQUIRE(image[0][0] == 1);
  REQUIRE(image[0][1] == 1);
  REQUIRE(image[0][2] == 1);
  REQUIRE(image[1][0] == 0);
  REQUIRE(image[1][1] == 0);
  REQUIRE(image[1][2] == 1);
  REQUIRE(image[2][0] == 0);
  REQUIRE(image[2][1] == 0);
  REQUIRE(image[2][2] == 1);

  naivebayes::label_t label = dataset.GetLabelAtIndex(0);
  REQUIRE(label == 1);
  label = dataset.GetLabelAtIndex(1);
  REQUIRE(label == 2);
  label = dataset.GetLabelAtIndex(2);
  REQUIRE(label == 7);
}
