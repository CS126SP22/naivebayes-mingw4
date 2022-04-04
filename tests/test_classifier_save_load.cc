#include <catch2/catch.hpp>

#include <core/dataset.h>
#include <core/classifier.h>
#include <core/classifier.h>
#include <string>
#include <sstream>
#include <limits>


TEST_CASE("Check classifier save and load") {

  naivebayes::ImageDataset dataset;
  naivebayes::Classifier classifier;

  // simulated dataset containing 3 images each of which has 3x3 size
  std::string test_images_set = "#  \n#  \n#  \n###\n #\n###\n###\n  #\n  #\n";
  // set their label as 1, 2, 7
  std::string test_labels_set = "1\n2\n7\n";

  std::istringstream ss(test_images_set);
  ss >> dataset;

  ss.clear();
  ss.str(test_labels_set);
  ss >> dataset;

  // fit the data
  classifier.Fit(dataset);

  // Save model
  classifier.SaveModel("_test_saved.model");

  naivebayes::Classifier new_classifier;

  // load model to the new_classifier
  new_classifier.LoadModel("_test_saved.model");


  // assert all parameters are the same
  REQUIRE(new_classifier.GetTrainedImageSize() == classifier.GetTrainedImageSize());

  std::vector<float> P_c = classifier.Get_P_c();
  std::vector<float> new_P_c = new_classifier.Get_P_c();
  for (int c = 0; c < 10; c++) {
    REQUIRE(P_c[c] == new_P_c[c]);
  }


  std::vector<std::vector<float>> P_x_eq_1_given_c = classifier.Get_P_x_eq_1_given_c();
  std::vector<std::vector<float>> P_x_eq_0_given_c = classifier.Get_P_x_eq_0_given_c();
  std::vector<std::vector<float>> new_P_x_eq_1_given_c = new_classifier.Get_P_x_eq_1_given_c();
  std::vector<std::vector<float>> new_P_x_eq_0_given_c = new_classifier.Get_P_x_eq_0_given_c();

  int N = classifier.GetTrainedImageSize();
  for (int c = 0; c < 10; c++) {
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        REQUIRE(P_x_eq_1_given_c[c][i*N + j] == Approx(new_P_x_eq_1_given_c[c][i*N + j]));
        REQUIRE(P_x_eq_0_given_c[c][i*N + j] == Approx(new_P_x_eq_0_given_c[c][i*N + j]));
      }
    }
  }

}
