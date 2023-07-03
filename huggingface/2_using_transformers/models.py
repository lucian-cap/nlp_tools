from transformers import BertConfig, BertModel

#Load the config file for BERT model and use it to instantiate the model
#   this model will have random weights
config = BertConfig()
model = BertModel(config)

#Examine the config for a given architecture
print(config)

#Load a models weights
model = BertModel.from_pretrained('bert-base-cased')

#Save a model in the given directory
model.save_pretrained('./2_using_transformers/saving_model_test/')