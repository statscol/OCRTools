
import logging
import supervision as sv
import cv2
import numpy as np
from collections import defaultdict
from pathlib import Path
from pdf2image import convert_from_path
from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology
from PIL import Image
from paddleocr import PaddleOCR
import click


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

class MultiStepPDFParser:

  def __init__(self,labels:dict[str],text_cls_id:int=0,**ocr_engine_kwargs):
    self.ocr_engine = PaddleOCR(use_angle_cls=True,**ocr_engine_kwargs)
    self.labels=labels
    self.text_cls_id=text_cls_id
    self.detector = GroundingDINO(ontology=CaptionOntology(self.labels),box_threshold=0.35,text_threshold=0.25)
    self.logger = logging.getLogger(__name__)

  def detect(self,image:np.ndarray|Image.Image):
    """use zero-shot to detect bboxes"""
    return self.detector.predict(np.flip(image.copy(),-1) if isinstance(image,np.ndarray) else image.copy())

  def preprocess(self,pdf_filepath:str,dpi:int=500):
    """runs preprocesing on pdf files -> pdf2image"""
    return convert_from_path(pdf_filepath, dpi)

  def get_text(self,img:np.ndarray):
    texts=[]
    try:
      result=self.ocr_engine.ocr(img,cls=True)[0]
      texts=[line[1][0] for line in result]
    except Exception as e:
      self.logger.info(f"Exception: {e}")
    return " ".join(texts)


  @staticmethod
  def crop_pred(image:np.ndarray | Image.Image,predictions:sv.detection.core.Detections,subset_ids:list[int]=None):
    """
    crop and save detections from the zero-shot-model, predictions in absolute format xyxy
    you can use subset_ids to include only the class_ids you want to crop
    """
    subset_ids=set(predictions.class_id.tolist()) if subset_ids is None else set(subset_ids)

    if isinstance(image,Image.Image):
      #make sure we get BGR images in numpy
      image=np.flip(np.asarray(image),-1)
    if len(predictions)>0:
      crops=[image[yi:yf,xi:xf] for idx,(xi,yi,xf,yf) in enumerate(predictions.xyxy.astype(int)) if predictions.class_id[idx] in subset_ids]
      return crops
    else:
      return

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
      detections=self.detect(page)
      images=self.crop_pred(page,detections)
      text=""
      for idx_img,img in enumerate(images):
        if save_images:
          cv2.imwrite(f"{output_path}/{Path(pdf_filepath).name.split('.')[0]}_image_{idx_img}.jpg",img)
        text+=self.get_text(img) if detections.class_id[idx_img]==self.text_cls_id else ""
      results[idx_page]['images']=images
      results[idx_page]['types']=detections.class_id
      results[idx_page]['text']=text
    return results

LABELS={"text chunk, paragraph or section with just text": "text","image or figure or diagram":"image-diagram","table":"table"}
processor=MultiStepPDFParser(labels=LABELS,**{'lang':'en','use_gpu':True}) ##if using cuda,add 'use_gpu':True

@click.command()
@click.option('--pdf_file',type=str,help='Path to pdf')
@click.option('--save_images', is_flag=True,help="Whether or not to save images to disk")
@click.option('--output_path', type=str,help="Path where images will be saved")
def process_pdf(pdf_file:str,save_images:bool,output_path:str):
  return processor.process(pdf_file,output_path,save_images)

if __name__=="__main__":
     ##if using cuda,add 'use_gpu':True
    results=process_pdf.main(standalone_mode=False)
    for k,v in results.items():
        print(f"Page {k}: {v['text']}")