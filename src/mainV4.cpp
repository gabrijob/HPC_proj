#include <chrono>
#include "CatDogCNNV2.h"
#include <omp.h>
using namespace std;
using namespace chrono;
using namespace tensorflow;
using namespace tensorflow::ops;

int k =10;

int main(int argc, const char * argv[])
{
    int image_side = 150;
    int image_channels = 3;
    int filter_side = 3;
    CatDogCNN model(image_side, image_channels); // CREATE CLONE METHOD
    Status s = model.CreateGraphForImage(true);
    TF_CHECK_OK(s);

    /* READ DATA */
    string base_folder = "../data/k_fold_data";
    int batch_size = 20;
    //Label: cat=0, dog=1
    vector<pair<Tensor, float>> all_files_tensors;
    s = model.ReadFileTensors(base_folder, {make_pair("cats", 0), make_pair("dogs", 1)}, all_files_tensors);
    TF_CHECK_OK(s);
    size_t nb_files = all_files_tensors.size();
    printf("Number of files: %lu\n", nb_files);
    assert(nb_files > 0);

    // Divide data in k folds
    vector <vector<int>> folds(k);
    for(int i = 0; i < nb_files; i++){
        folds[i%k].push_back(i);
    }
 
    vector<pair<Tensor, float>> train_tensors, validation_tensors, test_tensors;
    float sum_acc = 0.0;

    /* CROSS VALIDATION */
    auto t_begin = high_resolution_clock::now();
    #pragma omp parallel num_threads(4) private(train_tensors, validation_tensors, test_tensors)
    {
        int id, nthr;
        id = omp_get_thread_num();
        nthr = omp_get_num_threads();
        int test_idx, valid_idx;
        float sum_acc_thr = 0.0;
        
        #pragma omp parallel for reduction (+:sum_acc_thr)
            for (test_idx = (k-1) - id; test_idx >= 0; test_idx -= nthr) 
            {            
                auto t_begin_cv = high_resolution_clock::now();
                CatDogCNN model_k(image_side, image_channels);
                Status s = model_k.CreateGraphForImage(true); 
                
                // Set the validation index fold
                if(test_idx > 0)
                    valid_idx = test_idx - 1;
                else
                    valid_idx = k-1;

                for(int i = 0; i < k; i++){
                    if(i == valid_idx) {
                        for(int j=0; j < folds[valid_idx].size(); j++)
                            validation_tensors.push_back(all_files_tensors[folds[i][j]]);
                    }
                    if(i == test_idx) {
                        for(int j=0; j < folds[test_idx].size(); j++)
                            test_tensors.push_back(all_files_tensors[folds[i][j]]);
                    }
                    else {
                        for(int j=0; j < folds[i].size(); j++)
                            train_tensors.push_back(all_files_tensors[folds[i][j]]);
                    }
                }

                // Create batches for training
                vector<Tensor> train_images, train_labels, valid_images, valid_labels;
                s = model_k.CreateBatches(train_tensors, batch_size, train_images, train_labels);
                TF_CHECK_OK(s);

                s = model_k.CreateBatches(validation_tensors, batch_size, valid_images, valid_labels);
                TF_CHECK_OK(s);

                // CNN model
                s = model_k.CreateGraphForCNN(filter_side);
                TF_CHECK_OK(s);
                s = model_k.CreateOptimizationGraph(0.0001f);//input is learning rate
                TF_CHECK_OK(s);

                //Run inititialization
                s = model_k.Initialize();
                TF_CHECK_OK(s);

                size_t num_batches = train_images.size();
                assert(num_batches == train_labels.size());
                size_t valid_batches = valid_images.size();
                assert(valid_batches == valid_labels.size());

                int num_epochs = 1; //hyperparameter
                //Epoch / Step loops
                for(int epoch = 0; epoch < num_epochs; epoch++)
                {
                    /* TRAINING */
                    cout << "Epoch " << epoch+1 << "/" << num_epochs << ":" <<  "[Iteration " << k-test_idx << "]\n";
                    auto t1 = high_resolution_clock::now();
                    float loss_sum = 0;
                    float accuracy_sum = 0;                
                    for(int b = 0; b < num_batches; b++)
                    {
                        vector<float> results;
                        float loss;
                        float accuracy;
                        s = model_k.TrainCNN(train_images[b], train_labels[b], results, loss);
                        loss_sum += loss;
                        accuracy = accumulate(results.begin(), results.end(), 0.f) / results.size();
                        accuracy_sum += accuracy;
                        //cout << ".";
                    }
                    /* VALIDATION */
                    //cout << endl << "Validation:\n";
                    float validation_sum = 0;
                    #pragma omp parallel for reduction (+:validation_sum)
                        for(int c = 0; c < valid_batches; c++)
                        {
                            vector<float> results;
                            s = model_k.ValidateCNN(valid_images[c], valid_labels[c], results);
                            validation_sum += accumulate(results.begin(), results.end(), 0.f) / results.size();
                            //cout << ".";
                        }
                    auto t2 = high_resolution_clock::now();
                    cout << "Time: " << duration_cast<seconds>(t2-t1).count() << " seconds";
                    cout << "Loss: " << loss_sum/num_batches << " Results accuracy: " << accuracy_sum/num_batches << " Validation accuracy: " << validation_sum/valid_batches << " [Iteration " << k-test_idx << "]" << endl;
                }

                /* TESTING */
                s = model_k.CreateGraphForImage(false);//rebuild the model without unstacking
                TF_CHECK_OK(s);

                vector<pair<Tensor,float>> stacked_test_tensors;
                s = model_k.OneBatch(test_tensors, stacked_test_tensors);
                TF_CHECK_OK(s);
                //test a few images
                size_t nb_test_files = stacked_test_tensors.size();
                int count_success = 0;
                for(int i = 0; i < nb_test_files; i++)
                {
                    pair<Tensor, float> p = stacked_test_tensors[i];
                    int result;
                    s = model_k.Predict(p.first, result);
                    TF_CHECK_OK(s);
                    //cout << "Test number: " << i + 1 << " predicted: " << result << " actual is: " << p.second << endl;
                    if(result == (int)p.second)
                        count_success++;
                }
                auto t_end_cv = high_resolution_clock::now();
                cout << "Total successes: " << count_success << " out of " << nb_test_files << " in " << duration_cast<seconds>(t_end_cv-t_begin_cv).count() << " seconds " << " [Iteration " << k-test_idx << "]" << endl;
                sum_acc_thr += float(count_success)/nb_test_files;

                // Forget current folds
                train_tensors.clear();
                validation_tensors.clear();
                test_tensors.clear();
                stacked_test_tensors.clear();   
            }

        #pragma omp critical
        {
            sum_acc += sum_acc_thr;
        }   
    }
    auto t_end = high_resolution_clock::now();
    cout << "Final average accuracy : " << sum_acc/k << "\n";
    cout << "Total time : " << duration_cast<seconds>(t_end-t_begin).count() << " seconds" << endl;

    return 0;
}
