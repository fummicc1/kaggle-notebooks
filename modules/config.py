class Config:    
    batch_size = 64
    num_workers = 8
    n_epochs = 15
    lr = 1e-5
    model_name = "vit_large_patch16_224"
    is_kaggle_notebook = True
    resized_height = 300    
    resized_width = 600
    
    base_input_path_for_kaggle = "/kaggle/input/cassava-leaf-disease-classification"
    base_input_path_for_local = "./"
    
    @property
    def base_input_path(self):
        return self.base_input_path_for_kaggle if self.is_kaggle_notebook else self.base_input_path_for_local
