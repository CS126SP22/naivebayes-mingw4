#include <core/dataset.h>
#include <core/classifier.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <limits>


namespace naivebayes {

Classifier::Classifier() {
  for (int c = 0; c < 10; c++) {
    P_c.push_back(0.0f);
  }
  trained_image_size = 0;
  laplacian_smooth_K = 1.0f;
}

void Classifier::Fit(ImageDataset &dataset) {
  
  int N = dataset.GetImageSize();
  trained_image_size = N;


  for (int c = 0; c < 10; c++) {
    std::vector<float> P_x_eq_0_given_ci;
    std::vector<float> P_x_eq_1_given_ci;
    for (int i = 0; i < N*N; i++) {
      P_x_eq_0_given_ci.push_back(0.0f);
      P_x_eq_1_given_ci.push_back(0.0f);
    }
    P_x_eq_0_given_c.push_back(P_x_eq_0_given_ci);
    P_x_eq_1_given_c.push_back(P_x_eq_1_given_ci);
  }


  for (int i = 0; i < dataset.GetDatasetSize(); i++) {
    image_t image = dataset.GetImageAtIndex(i);
    label_t label = dataset.GetLabelAtIndex(i);

    P_c[label] += 1.0f;

    for (int _i = 0; _i < N; _i ++) {
      for (int _j = 0; _j < N; _j ++) {
        if (image[_i][_j] == 0) {
          P_x_eq_0_given_c[label][_j +_i*N] += 1.0f;
        } else {
          P_x_eq_1_given_c[label][_j +_i*N] += 1.0f;
        }
      }
    }
  }

  for (int c = 0; c < 10; c++) {
    for (int _i = 0; _i < N; _i ++) {
      for (int _j = 0; _j < N; _j ++) {
        P_x_eq_0_given_c[c][_j +_i*N] = (P_x_eq_0_given_c[c][_j +_i*N] + laplacian_smooth_K) / (P_c[c] + laplacian_smooth_K * 2.0f);
        P_x_eq_1_given_c[c][_j +_i*N] = (P_x_eq_1_given_c[c][_j +_i*N] + laplacian_smooth_K) / (P_c[c] + laplacian_smooth_K * 2.0f);
      }
    }

    P_c[c] = (P_c[c] + laplacian_smooth_K) / (dataset.GetDatasetSize() + laplacian_smooth_K * 1.0f);
  }
}

label_t Classifier::Predict(image_t &image) {

  if (trained_image_size == 0) {
    std::cout << "Error: Classifier not trained!" << std::endl;
    return -1;
  }

  int N = image.size();
  if (N != trained_image_size) {
    std::cout << "Error: Image size does not match the model." << N << " and " << trained_image_size << "." << std::endl;
    return -1;
  }

  label_t predict_class = -1;
  float max_log_prob = -std::numeric_limits<double>::infinity();

  for (int c = 0; c < 10; c ++) {
    float log_prob_c = logf( P_c[c] );
    for (int _i = 0; _i < N; ++_i) {
      for (int _j = 0; _j < N; ++_j) {
        unsigned char pixel = image[_i][_j];
        if (pixel == 0)
          log_prob_c += logf( P_x_eq_0_given_c[c][_j + _i*N] );
        else
          log_prob_c += logf( P_x_eq_1_given_c[c][_j + _i*N] );
      }
    }

    if (log_prob_c > max_log_prob) {
      max_log_prob = log_prob_c;
      predict_class = c;
    }
  }
  return predict_class;
}


void Classifier::SaveModel(std::string save_path) {
  std::ofstream ofs(save_path);

  ofs << trained_image_size << std::endl;
  for (int c = 0; c < 10; c++) {
    ofs << P_c[c] << std::endl;
  }

  for (int c = 0; c < 10; c++) {
    for (int i = 0; i < trained_image_size*trained_image_size; i++) {
      ofs << P_x_eq_0_given_c[c][i] << std::endl;
    }
  }

  for (int c = 0; c < 10; c++) {
    for (int i = 0; i < trained_image_size*trained_image_size; i++) {
      ofs << P_x_eq_1_given_c[c][i] << std::endl;
    }
  }

  ofs.close();
}


void Classifier::LoadModel(std::string load_path) {
  std::ifstream ofs(load_path);

  ofs >> trained_image_size;
  for (int c = 0; c < 10; c++) {
    ofs >> P_c[c];
  }

  int N = trained_image_size;
  P_x_eq_0_given_c.empty();
  P_x_eq_1_given_c.empty();

  for (int c = 0; c < 10; c++) {
    std::vector<float> P_x_eq_0_given_ci;
    std::vector<float> P_x_eq_1_given_ci;
    for (int i = 0; i < N*N; i++) {
      P_x_eq_0_given_ci.push_back(0.0f);
      P_x_eq_1_given_ci.push_back(0.0f);
    }
    P_x_eq_0_given_c.push_back(P_x_eq_0_given_ci);
    P_x_eq_1_given_c.push_back(P_x_eq_1_given_ci);
  }


  for (int c = 0; c < 10; c++) {
    for (int i = 0; i < trained_image_size*trained_image_size; i++) {
      ofs >> P_x_eq_0_given_c[c][i];
    }
  }

  for (int c = 0; c < 10; c++) {
    for (int i = 0; i < trained_image_size*trained_image_size; i++) {
      ofs >> P_x_eq_1_given_c[c][i];
    }
  }

  ofs.close();
}

std::vector<float> &Classifier::Get_P_c() {
  return P_c;
}

std::vector<std::vector<float>> &Classifier::Get_P_x_eq_0_given_c() {
  return P_x_eq_0_given_c;
}

std::vector<std::vector<float>> &Classifier::Get_P_x_eq_1_given_c() { 
  return P_x_eq_1_given_c;
}

int Classifier::GetTrainedImageSize () {
  return trained_image_size;
}

}  // namespace naivebayes
