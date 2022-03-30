#include <iostream>
#include <fstream>
#include <cstring>
#include <core/dataset.h>
#include <core/classifier.h>

int main(int argc, char* argv[]) {

  std::string train_data_path = "";
  std::string save_model_path = "";

  // Paring the arguments
  if (argc != 5) {
    printf("Unrecognized command.\n");
    printf("Please run like this\n\t./train-model train train_data_path save save_model_path\n");
    return -1;
  }

  for (int i = 1; i <= argc - 1; i += 2) {
    if (strcmp(argv[i], "train") == 0) {
      train_data_path = std::string (argv[i + 1]);
    }
    else if (strcmp(argv[i], "save") == 0) {
      save_model_path = std::string (argv[i + 1]);
    }
    else {
      printf("Unrecognized flag: %s.\n", argv[i]);
      printf("Please run like this\n\t./train-model train train_data_path save save_model_path\n");
      return -1;
    }
  }

  if (train_data_path == "" || save_model_path == "") {
    printf("Unrecognized command.\n");
    printf("Please run like this\n\t./train_model train train_data_path save save_model_path\n");
    return -1;
  }


  naivebayes::ImageDataset dataset(train_data_path);
  // dataset.PrintInfo();

  naivebayes::Classifier classifier;

  classifier.Fit(dataset);

  classifier.SaveModel(save_model_path);

/*  // Testing
  classifier.LoadModel("../data/saved_model");

  for (int i = 0; i < dataset.GetDatasetSize(); i++) {
    naivebayes::image_t image = dataset.GetImageAtIndex(i);
    naivebayes::label_t label = dataset.GetLabelAtIndex(i);

    naivebayes::label_t predict = classifier.Predict(image);

    std::cout << i << ", " << label << ", " << predict << std::endl;
    }*/

  return 0;
}
