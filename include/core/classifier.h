#pragma once

#include <string>
#include <vector>
#include "dataset.h"

namespace naivebayes {

// using image_t = std::vector <std::vector <unsigned char> >;
// using label_t = int;

class Classifier {
 public:
  /**
   * Constructs a default naivebayes classifier.
   */
  Classifier();

  /**
   * Fit the given dataset.
   * @param dataset The given image dataset to be fitted.
   */
  void Fit(ImageDataset &dataset);

  /**
   * Predict the given image class.
   * @param image The given image to be predicted.
   * @return the predicted class.
   */
  label_t Predict(image_t &image);

  /**
   * Validate the given test dataset.
   * @param test_dataset The given image test dataset to be Validated.
   * @return the predicted accuracy.
   */
  float Validate(ImageDataset &test_dataset);

  /**
   * Save model parameters.
   * @param save_path the saved model path
   */
  void SaveModel(std::string save_path);

  /**
   * Load model parameters.
   * @param load_path the loaded model path
   */
  void LoadModel(std::string load_path);

  // Getters for naivebayes classifier.
  int GetTrainedImageSize();
  std::vector<float> &Get_P_c();
  std::vector<std::vector<float>> &Get_P_x_eq_0_given_c();
  std::vector<std::vector<float>> &Get_P_x_eq_1_given_c();

 private:

  // Configurations of naivebayes classifier.
  int trained_image_size;
  float laplacian_smooth_K;

  // Parameters of naivebayes classifier.
  std::vector<float> P_c;                               // P(c)
  std::vector<std::vector<float>> P_x_eq_0_given_c;     // P(x_i==0|c)
  std::vector<std::vector<float>> P_x_eq_1_given_c;     // P(x_i==1|c)
};

}  // namespace naivebayes
