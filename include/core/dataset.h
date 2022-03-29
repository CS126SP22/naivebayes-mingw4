#pragma once

#include <string>
#include <vector>

namespace naivebayes {

using image_t = std::vector <std::vector <unsigned char> >;
using label_t = int;

class ImageDataset {
 public:
  /**
   * Constructs a default image dataset.
   */
  ImageDataset();

  /**
   * Constructs a default image dataset.
   * @param data_path the path of dataset file.
   */
  ImageDataset(std::string data_path);

  // Getters for the dataset informations.
  int GetDatasetSize() const;
  int GetImageSize() const;
  image_t &GetImageAtIndex(int index);
  label_t  GetLabelAtIndex(int index);

  // Add a new image or label in the dataset.
  void AppendImage(image_t &new_image);
  void AppendLabel(label_t new_label);

  // Print dataset info for debug
  void PrintInfo();

  // The >> operater overriding for reading data from streams
  friend std::istream& operator>>(std::istream& in, ImageDataset& D);

 private:
  // The images data list.
  std::vector<image_t> images_data;

  std::vector<label_t> labels_data;

};

// The >> operater overriding for reading data from streams
std::istream& operator>>(std::istream& in, ImageDataset& D);

}  // namespace naivebayes

