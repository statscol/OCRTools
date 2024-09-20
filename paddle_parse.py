
import logging
from pdf2image import convert_from_path
import cv2
import numpy as np
from PIL import Image
from paddleocr import PPStructure
from collections import defaultdict
from pathlib import Path
import time
from datetime import timedelta
import click

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

class PaddlePDFParser:

  def __init__(self,**engine_kwargs):
    self.engine = PPStructure(lang="en",**engine_kwargs)
    self.logger = logging.getLogger(__name__)

  def preprocess(self,pdf_filepath:str,dpi:int=500):
    """runs preprocesing on pdf files -> pdf2image"""
    return convert_from_path(pdf_filepath, dpi)

  def get_text(self,layout_out:list):
    """
    join text found in chunks of the layout of type 'text' and 'title'
    """
    text=""
    for chunk in layout_out:
      if chunk['type'] in ("text","title"):
        text+=" ".join(txt['text'] for txt in chunk['res'])
    return text

  def process(self,pdf_filepath:str,output_path:str,save_images:bool=False):
    """
    runs the detection+ocr pipeline
    """
    ##TO DO: enable ProcessPoolExecutor to process every page in parallel if using cpu
    pages=self.preprocess(pdf_filepath)
    self.logger.info(f"Processing {len(pages)} Pages")
    results=defaultdict(defaultdict)
    Path(output_path).mkdir(parents=True,exist_ok=True)
    for idx_page,page in enumerate(pages):
      detections=self.engine(np.asarray(page))
      text=self.get_text(detections)
      images=[]
      if save_images:
        for idx_img,det in enumerate(detections):
          images.append(det['img'])
          cv2.imwrite(f"{output_path}/{Path(pdf_filepath).name.split('.')[0]}_image_page{idx_page+1}_{idx_img}.jpg",det['img'])
      results[idx_page]['images']=images
      results[idx_page]['tables']=[detection['res']['html'] for detection in detections if detection['type']=='table']
      results[idx_page]['text']=text
    return results

processor=PaddlePDFParser(**{'use_gpu':True})


@click.command()
@click.option('--pdf_file',type=str,help='Path to pdf')
@click.option('--save_images', is_flag=True,help="Whether or not to save images to disk")
@click.option('--output_path', type=str,help="Path where images will be saved")
def process_pdf(pdf_file:str,save_images:bool,output_path:str):
  return processor.process(pdf_file,output_path,save_images)
  
if __name__=="__main__":
    # if using cuda,add 'use_gpu':True
    # results contain per page index the following keys: text, images and tables
    results=process_pdf.main(standalone_mode=False)
    for k,v in results.items():
        print(f"Page {k}: {v['text']}")

