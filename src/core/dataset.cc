#include <core/dataset.h>
#include <iostream>
#include <fstream>

namespace naivebayes {

ImageDataset::ImageDataset() {

}

ImageDataset::ImageDataset(std::string data_path, bool train_flag) {

  if (train_flag) {
    std::string images_data_file = data_path + "/" + "trainingimages";
    std::string labels_data_file = data_path + "/" + "traininglabels";

    // Loading images data
    std::ifstream infile;
    infile.open(images_data_file);
    infile >> *this;
    infile.close();

    // Loading labels data
    infile.open(labels_data_file);
    infile >> *this;
    infile.close();
  } else {
    // Loading data
    std::ifstream infile;
    infile.open(data_path);
    infile >> *this;
    infile.close();
  }

}

int ImageDataset::GetDatasetSize() const {
  return images_data.size();
}

int ImageDataset::GetImageSize() const {
  return images_data[0].size();
}

image_t &ImageDataset::GetImageAtIndex(int index) {
  return images_data[index];
}

label_t ImageDataset::GetLabelAtIndex(int index) {
  return labels_data[index];
}

void ImageDataset::AppendImage(image_t &new_image) {
  images_data.push_back(new_image);
}

void ImageDataset::AppendLabel(label_t new_label) {
  labels_data.push_back(new_label);
}

void ImageDataset::PrintInfo() {
  std::cout << ">>>  Dataset info  <<<" << std::endl;
  std::cout << "Images number: " << images_data.size() << std::endl;
  std::cout << "Labels number: " << labels_data.size() << std::endl;
  if (images_data.size())
  std::cout << "Image's size: " << images_data[0].size() << "x" << images_data[0][0].size() << std::endl;
  std::cout << ">>> Dataset info end <<<" << std::endl;
}

std::istream& operator>>(std::istream& in, ImageDataset& D) {

  while (true) {

    std::vector<unsigned char> first_row;
    char x;

    // Read the first row and get the image size
    while (true) {
      if (in.eof()) return in;
      in.get(x);
      if (x == '\n') break;
      else first_row.push_back(x);
    }

    int row_size = first_row.size();

    // If the row size is 0, the end
    if (row_size == 0) {
      return in; 
    }
    // If the row size is 1, it is a label
    else if (row_size == 1) {
      D.AppendLabel( (label_t) (first_row[0] - '0'));
    }
    // Else it is an image row, then read the left row_size-1 rows
    else {

      image_t _image;

      for (int _i = 0; _i < row_size; ++_i) {
        first_row[_i] = first_row[_i] == ' ' ? 0 : 1;    // not shaded or shaded
      }

      _image.push_back(first_row);

      for (int i = 0; i < row_size-1; ++i) {
        std::vector<unsigned char> new_row;
        while (true) {

          if (in.eof()) return in;

          in.get(x);
          if (x == '\n') break;
          else
            new_row.push_back(
              x == ' ' ? 0 : 1    // not shaded or shaded
            );
        }

        _image.push_back(new_row);
      }

      // Append the image into the dataset
      D.AppendImage(_image);
    }
  }

  return in;
}


}  // namespace naivebayes