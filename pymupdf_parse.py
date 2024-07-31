from collections import defaultdict
from pathlib import Path
import fitz
from PIL import Image
import io
import click


# Based and edited from PyMuPDF docs and https://aliarefwriorr.medium.com/extract-all-images-from-pdf-in-python-cda3dc195abd

@click.command()
@click.option('--pdf_file',type=str,help='Path to pdf')
@click.option('--save_images', is_flag=True,help="Whether or not to save images to disk")
@click.option('--output_path', type=str,help="Path where images will be saved")
def pdf2data(
    pdf_file:str,
    save_images:bool= False,
    output_path:str = None,
    **table_detection_kwargs):
  """
  Parse a pdf file and returns text, images and tables separately in a dictionary.
  """
  Path(output_path).mkdir(parents=True, exist_ok=True)
  assert Path(pdf_file).exists(),"File does not exist"

  with fitz.open(pdf_file) as doc:
    pages=defaultdict(defaultdict)
    for page_index in range(len(doc)):
        tables,images=[],[]
        # get the page itself
        page = doc[page_index]
        text= page.get_text()
        pages[page_index]['text'] = text
        tabs = page.find_tables(**table_detection_kwargs)
        if tabs:
          print(f"[+] Found a total of {len(tabs.tables)} tables in page {page_index}")
          for t in tabs:
            #get table in markdown format
            tables.append(t.to_markdown())
        pages[page_index]['tables']=tables
        image_list = page.get_images()
        # printing number of images found in this page
        if image_list:
            print(f"[+] Found a total of {len(image_list)} images in page {page_index}")

            for image_index, img in enumerate(page.get_images(), start=1):
                # get the XREF of the image
                xref = img[0]
                # extract the image bytes
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                # get the image extension
                image_ext = base_image["ext"]
                # load it to PIL
                image = Image.open(io.BytesIO(image_bytes))
                # save it to local disk
                images.append(image)
                if save_images:
                  image.save(open(f"{output_path}/{Path(pdf_file).name.split('.')[0]}image{page_index+1}_{image_index}.{image_ext}", "wb"))
        pages[page_index]['images']=images

    return pages

if __name__=="__main__":
    results=pdf2data.main(standalone_mode=False)
    print(results[0]['text'])
    print(results[0]['tables'])
    