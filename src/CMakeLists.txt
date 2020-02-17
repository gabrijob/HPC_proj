target_sources(CatDogCNN_simple 
	PRIVATE 
		${CMAKE_CURRENT_LIST_DIR}/main_simple.cpp 
		${CMAKE_CURRENT_LIST_DIR}/CatDogCNN.cpp 
	PUBLIC 
		${CMAKE_CURRENT_LIST_DIR}/CatDogCNN.h
)

target_sources(CatDogCNN_kfold_cv 
	PRIVATE 
		${CMAKE_CURRENT_LIST_DIR}/main_kfold_cv.cpp 
		${CMAKE_CURRENT_LIST_DIR}/CatDogCNN.cpp 
	PUBLIC 
		${CMAKE_CURRENT_LIST_DIR}/CatDogCNN.h
)