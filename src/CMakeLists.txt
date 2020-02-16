target_sources(CatDogCNN_simple 
	PRIVATE 
		${CMAKE_CURRENT_LIST_DIR}/mainV2.cpp 
		${CMAKE_CURRENT_LIST_DIR}/CatDogCNNV2.cpp 
	PUBLIC 
		${CMAKE_CURRENT_LIST_DIR}/CatDogCNNV2.h
)

target_sources(CatDogCNN_kfold_cv 
	PRIVATE 
		${CMAKE_CURRENT_LIST_DIR}/mainV4.cpp 
		${CMAKE_CURRENT_LIST_DIR}/CatDogCNNV2.cpp 
	PUBLIC 
		${CMAKE_CURRENT_LIST_DIR}/CatDogCNNV2.h
)