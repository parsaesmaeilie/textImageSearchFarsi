
from transformers import AutoTokenizer, CLIPModel  , AutoProcessor,MT5Tokenizer,MT5ForConditionalGeneration
from sklearn.metrics.pairwise import cosine_similarity as cs_sim
from torch.nn.functional import cosine_similarity
from sklearn.cluster import KMeans
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import numpy as np
import shutil
import torch
import json
import time
import os

#Load Model
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
model_size = "base"
model_name = f"persiannlp/mt5-{model_size}-parsinlu-opus-translation_fa_en"
tokenizer_translation = MT5Tokenizer.from_pretrained(model_name)
model_translate= MT5ForConditionalGeneration.from_pretrained(model_name)


def run_translation_model(input_string, **generator_args):
    print(input_string)
    input_ids = tokenizer_translation.encode(input_string, return_tensors="pt")
    res = model_translate.generate(input_ids, **generator_args)
    output = tokenizer_translation.batch_decode(res, skip_special_tokens=True)
    print(output)
    return str(output[0])
# set_file_path
file = open('./path.cfg','r+')
lines = file.read()
confing = json.loads(lines)
annotation_file = confing['anotations']
image_dir_root = confing['image_path']




class Clip_workFlow:
    def __init__(self,image_dir_root,processor,model,tokenizer,annotation_file):
        self.image_dir_root = image_dir_root
        self.processor = processor
        self.model = model
        self.tokenizer = tokenizer
        self.annotation_file = annotation_file
    def image_loader(self,image_ids_batch,coco):
        images =[]
        image_annotations = []
        for image_id in tqdm (image_ids_batch,desc="Processing", unit="item"):
            image_info = coco.loadImgs(image_id)[0]
            image_path = self.image_dir_root+image_info['file_name']
            images.append(Image.open(image_path))
            annotation_ids = coco.getAnnIds(imgIds=image_info['id'])
            annotations = coco.loadAnns(annotation_ids)
            image_annotations.append(','.join(list(set([self.category_names_by_id[i['category_id']] for i in annotations]))))
        return image_annotations,images
    def similarity_checker(self ,image_list,image_annotations):
        
        image_inputs = self.processor(images=image_list, return_tensors="pt", padding=True)
        image_feature = self.model.get_image_features(**image_inputs)
        tokenized_caps = self.tokenizer(image_annotations, padding=True, return_tensors="pt")
        text_feature = self.model.get_text_features(**tokenized_caps)
        similarity_matrix = cosine_similarity(text_feature.unsqueeze(1), image_feature.unsqueeze(0), dim=2)
        return similarity_matrix
    def save_images(self,embedde_image,name):
        torch.save(embedde_image,f'./{self.image_dir_root}/outputs/{name}.pt')
        name_files = open(f'{self.image_dir_root}_saved_image.txt','a+')
        name_files.write(f'{name}\n')
        name_files.close()
    def load_images(self,path):
        name_files = pd.read_csv(path)
        
        loaded_image = []
        for image_name in tqdm (name_files['image_name'],desc="Processing", unit="item"):
            embede_image = torch.load(f'./{self.image_dir_root}/outputs/{image_name.strip()}.pt')
            loaded_image.append(embede_image)
        return loaded_image
    def cluseter_images(self,image_feature):
        df = pd.read_csv('image_embding.csv')
        image_matrix = torch.cat(image_feature)
        print('start clustering')
        kmeans = KMeans(n_clusters=70 , n_init=10)
        cluster_assignments = kmeans.fit_predict(image_matrix.detach().numpy())
        print('save clusters')
        df['cluster_lable'] = cluster_assignments
        cluster_means  = kmeans.cluster_centers_
        np.save('cluster_means.npy', cluster_means)
        df.to_csv('image_embding.csv', index=False)
        print('done')
    def load_cluster(self,text_feature):
        cluster_means = np.load('cluster_means.npy')
        sims = []  
        cosine = cs_sim(text_feature.detach().numpy(), cluster_means )
        most_similar_index = np.argmax(cosine, axis=1)
        
        clussetr_label  = most_similar_index
        
        df = pd.read_csv('image_embding.csv')
        print(clussetr_label)
        images= df[df['cluster_lable']==clussetr_label[0]]['image_name']
        loaded_image = []
        print(len(images))
        for image_name in tqdm (images,desc="Processing", unit="item"):
            embede_image = torch.load(f'./outputs/{image_name.strip()}.pt')
            loaded_image.append(embede_image)
        return self.check_similarity_test2(loaded_image,text_feature)
        
    def make_it_csv(self):
        name_files = open('{self.image_dir_root}_saved_image.txt','r+')
        images_names = name_files.readlines()
        df = pd.DataFrame(images_names, columns=['image_name'])
        df['image_name'] = df['image_name'].str.replace('\n', '')
        df['file_path'] = f'./{self.image_dir_root}/outputs/'+df['image_name']
        df.to_csv(f'{self.image_dir_root}_image_embding.csv',index=False)
    def image_open_feature_save(self):
        coco = COCO(self.annotation_file)
        all_image_ids = coco.getImgIds()
        for image_id in tqdm (all_image_ids,desc="Processing", unit="item"):
            image_info = coco.loadImgs(image_id)[0]
            image_path = self.image_dir_root+image_info['file_name']
            current_image = Image.open(image_path)
            image_inputs = self.processor(images=current_image, return_tensors="pt", padding=True)
            image_feature = self.model.get_image_features(**image_inputs)
            self.save_images(image_feature,image_info['file_name'])
            current_image.close()  
    def ProcessImageDataSet(self,image_infos):
        for image_info in tqdm (image_infos,desc="Processing", unit="item"):
            image_path = self.image_dir_root+image_info
            current_image = Image.open(image_path)
            image_inputs = self.processor(images=current_image, return_tensors="pt", padding=True)
            image_feature = self.model.get_image_features(**image_inputs)
            self.save_images(image_feature,image_info)
            current_image.close()        

    def save_plot(self,title,similarity_matrix):
        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix.detach().numpy(), cmap="YlGnBu", cbar_kws={'label': 'Cosine Similarity'})
        plt.title(title)
        plt.savefig(f'./{title}.png')
    def process_cooc(self,batch_size=1000):
        coco = COCO(self.annotation_file)
        all_image_ids = coco.getImgIds()
        categories = coco.loadCats(coco.getCatIds())
        category_names_by_id = {category['id']: category['name'] for category in categories}
        batch_number = len(all_image_ids)//batch_size
        for i in range(batch_number):
            notes,imgs = self.image_loader( all_image_ids[i::batch_number],coco)
            self.save_plot(f'batch number {i+1}',self.similarity_checker(imgs,notes))
    def check_similarity_test2(self,image_embding,text_feature):
        sims = []   
        
        sims = list(enumerate(sims))
        sims.sort(key=lambda x :x[1],reverse= True)
        print( [sim[0] for sim in sims[:20] ])
        image_features= torch.cat(image_embding)
        text_features = text_feature / text_feature.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        similarities = (text_features @ image_features.T).squeeze()
        top10_indices = torch.argsort(similarities, descending=True)[:10]
        print(top10_indices)
        cosine = cs_sim(text_feature.detach().numpy(), torch.cat(image_embding).detach().numpy())
        most_similar_index = np.argsort(cosine, axis=1)[:, -50:]
        print(most_similar_index)
        return top10_indices.tolist()
    
    def check_similarity_test(self,queary,images_embded=None):
        tokenized_caps = self.tokenizer(queary, padding=True, return_tensors="pt")
        text_feature = self.model.get_text_features(**tokenized_caps)
        if images_embded == None:
            top_10_indices = self.load_cluster(text_feature)
        else:
            top_10_indices = self.check_similarity_test2(images_embded,text_feature)
        # similarity_matrix = cosine_similarity(text_feature.unsqueeze(1), images_embded.unsqueeze(0), dim=2)
        # top_10_indices = np.argsort(similarity_matrix.detach().numpy())[-10:][::-1]
        # #top_10_indices=  top_10_indices[:, :10]
        # top_10_indices = [top_indices[:, 0] for top_indices in  top_10_indices]

        return top_10_indices
    def saveimages_top_10(self,top_10,queary):
        name_files = pd.read_csv('image_embding.csv')
        
        queary= queary.replace(' ','_')
        path = f'./result/{queary}'
        os.makedirs(f'./result/{queary}' ,exist_ok = True)
        for i in top_10:
            shutil.copy(self.image_dir_root+(name_files.iloc[i])['image_name'],f'{path}/')
        print(f'save {len(top_10)} in {path} ')
    
checker = Clip_workFlow(image_dir_root,processor,model,tokenizer,annotation_file)


images = checker.load_images('image_embding.csv')

print('starting')
while True:
    try:
        query  = run_translation_model(input())
        start_time = time.time()
        top_10= checker.check_similarity_test(query,images)
        print(f'execution time was {time.time()-start_time}')
        checker.saveimages_top_10(top_10,query)
    except KeyboardInterrupt:
        print('bye')
        break
    except Exception as e:
        print('somtihng wrong happend')
        log_file = open('error.log','a+')
        log_file.writelines(e)


