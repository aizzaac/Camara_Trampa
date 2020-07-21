"""
Created on Tue May 19 18:23:25 2020
​
@author: aizzaac
"""


"""
* snapshot.py se llama como un subproceso: Se toma una foto usando la techa de "espacio", y se sale del ambiente con "q".
* Muchos modelos de clasificación de imágenes van a ejecutarse y hacer inferencia sobre la foto. Actualmente hay 9 modelos. Todos ellos han sido entrenados con la misma
  base de datos (ImageNet), pero usan diferente arquitectura de Aprendizaje Profundo (https://coral.ai/models/).
* La siguiente información, por cada modelo IA, va a ser guardada en una lista (Imagen, Inferencia, Etiqueta, Modelo, Puntaje, Temperatura del TPU, Tiempo(ms)). 
* Solamente el modelo IA con el mejor Puntaje será indexado en Elasticsearch.
* La foto tomada (img0000.jpg) será borrada.
****snapshot.py, classify.py se necesitan para ejecutar este código
"""


import re 
import os 
from pathlib import Path
import time

from PIL import Image  

import classify  
import tflite_runtime.interpreter as tflite  

import subprocess 


EDGETPU_SHARED_LIB = 'libedgetpu.so.1'  # EDGETPU delegate for Linux
count = 2 
top_k = 1 
threshold = 0.0

rootdir = os.getcwd() 
dir_path= Path(rootdir) 


dummy = [] 
groups = [] 



import elasticsearch6  
from elasticsearch6 import Elasticsearch, helpers
import datetime


ES_SERVER = "http://172.16.1.132:9200/" 
INDEX_NAME = "coral_ia" 
DOC_TYPE = 'coral_edge'  



"""
To read ".txt" files which contain the labels (works for all Models) 
"""
def load_label(path):
    with open(path, 'r', encoding='utf-8') as f: 
        lines = f.readlines() 
    ret = {}  
    for row_number, content in enumerate(lines): 
        pair = re.split(r'[:\s]+', content.strip(), maxsplit=1) 
        if len(pair) == 2 and pair[0].strip().isdigit():
            ret[int(pair[0])] = pair[1].strip()
        else: 
            ret[row_number] = pair[0].strip()



"""
To run inference after: converting model to .tflite, doing quantization and compilation
"""
def make_interpreter(model_file):
  model_file, *device = model_file.split('@')
  return tflite.Interpreter(
      model_path=model_file,
      experimental_delegates=[
          tflite.load_delegate(EDGETPU_SHARED_LIB,
                               {'device': device[0]} if device else {})
      ])



"""
To load ".tflite" files. These are the AI models
"""
def getInterpreter(root, files):
    for file in files:
        filepath = os.path.join(root, file)
        if filepath.endswith(".tflite"):
            dummy.append(f'Model:{file}')
            print("Model:", file)
            print("\n")
            interpreter = make_interpreter(filepath)
            interpreter.allocate_tensors()
            return interpreter   
    return None



"""
To load ".jpg, .bmp or .png files. This is the taken photo
"""
def getImage(dir_path, image_file):
    for file in image_file:
         if re.match('.*\.jpg|.*\.bmp|.*\.png', file): 
                filepath = os.path.join(dir_path, file)
                dummy.append(f'Image:{file}')
                print("Image:", file)
                print("\n")
                return filepath
    return None
        


"""
To load ".txt" files. These are the "labels"
"""
def getLabel(root, files):
    for file in files:
        filepath = os.path.join(root, file)
        if filepath.endswith(".txt"):
            dummy.append(f'Labels:{file}')
            print("Labels:", file)
            print("\n")
            return load_label(filepath)
    return None



"""
To read the temperature of the TPU 
"""
def check_temperature_status():
    try:
        with open('/sys/devices/platform/33c00000.pcie/pci0001:00/0001:00:00.0/0001:01:00.0/apex/apex_0/temp') as f:
            status = f.read()
            print("TPU_temp(°C):", int(status)/1000)
            return status
    except FileNotFoundError:
        pass



"""
To sort the AI model with the best "Score" 
"""
def max_group():
    for v in dummy: 
        if v.startswith('Labels:'):
            groups.append([v]) 
        else:
            groups[-1].append(v) 
    max_group = max(groups, key=lambda k: float(re.search(r'Score:([\d.]+)', ' '.join(k)).group(1)))
    return max_group




"""
Initialize Elasticsearch by server's IP'
"""
def initialize_elasticsearch():
    n = 0  
    while n <= 10: 
        try:
            es = Elasticsearch(ES_SERVER) 
            print("Initializing Elasticsearch...")
            return es
        except elasticsearch6.exceptions.ConnectionTimeout as err: 
            print(err)
            n += 1 
            continue
    raise Exception
    
    
    

"""
Create an index in Elasticsearch if one isn't already there
Create a mapping for every field, otherwise Elasticsearch will do it
"""
def initialize_mapping(es):
    mapping_classification = {
        'properties': {
            '@timestamp': {'type': 'date'},
            'Labels': {'type': 'keyword'},
            'Model': {'type': 'keyword'},
            'Image': {'type': 'keyword'},
            'Time(ms)': {'type': 'short'},
            'Inference': {'type': 'text'}, 
            'Score': {'type': 'short'},
            'TPU_temp(°C)': {'type': 'short'}
        }
    }
    print("Initializing the mapping ...")  
    if not es.indices.exists(INDEX_NAME): 
        es.indices.create(INDEX_NAME) 
        es.indices.put_mapping(body=mapping_classification, doc_type=DOC_TYPE, index=INDEX_NAME) 
        



def main():
    subprocess.run('/usr/bin/snapshot', shell=False)       
    image_file = os.listdir(rootdir) 
    
    for root, subdirs, files in os.walk(rootdir):

        labels = getLabel(root, files)

        interpreter = getInterpreter(root, files)
                
        if interpreter is not None:
            size = classify.input_size(interpreter)
            
            image_path = getImage(dir_path, image_file)
            
            image = Image.open(image_path).convert('RGB').resize(size, Image.ANTIALIAS)
    
            classify.set_input(interpreter, image)
    
            print('*The first inference on Edge TPU is slow because it includes',
                  'loading the model into Edge TPU memory*')
            for _ in range(count):
                start = time.perf_counter()
                interpreter.invoke()
                inference_time = time.perf_counter() - start
                classes = classify.get_output(interpreter, top_k, threshold)
                dummy.append(f'Time(ms):{(inference_time*1000):.4}')
                print('Time(ms):', '%.1f' % (inference_time * 1000))
            print("\n")   
                
            for klass in classes:
                dummy.append(f'Inference:{(labels.get(klass.id, klass.id))}')
                print('Inference:', '%s' % (labels.get(klass.id, klass.id)))
                dummy.append(f'Score:{(klass.score):.5}')
                print('Score:', '%.5f' % (klass.score))
                print("\n")
    
    
    maX_group = max_group() 
      
    temperature = check_temperature_status()
    maX_group.append(f'TPU_temp(°C):{int(temperature)/1000}')
    print('#####################################')
    print("\n")
    
   
    

    es=initialize_elasticsearch() 
    initialize_mapping(es)    


    actions = [
        {
            '_index': INDEX_NAME,
            '_type': DOC_TYPE,
            "@timestamp": str(datetime.datetime.utcnow().strftime("%Y-%m-%d"'T'"%H:%M:%S")),
            "Labels": maX_group[0].split(":")[1],
            "Model": maX_group[1].split(":")[1],
            "Image": maX_group[2].split(":")[1],
            "Time(ms)": maX_group[4].split(":")[1],
            "Inference": maX_group[5].split(":")[1],
            "Score": maX_group[6].split(":")[1],
            "TPU_temp(°C)": maX_group[7].split(":")[1]
        
        }]

    try:
        res=helpers.bulk(client=es, index = INDEX_NAME, actions = actions)
        print ("\nhelpers.bulk() RESPONSE:", res)
        print ("RESPONSE TYPE:", type(res))
        
    except Exception as err: 
        print("\nhelpers.bulk() ERROR:", err)
    
    print("\n")
    print("\n")
    
    os.remove(image_path)
    print("Photo has been deleted")


if __name__ == "__main__":
    main()
    
