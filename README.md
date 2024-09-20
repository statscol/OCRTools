# OCR Tools for parsing PDF Files

This repo contains tools for parsing pdf files and its content using open source frameworks and models.

# Setup 

- Using a virtual env of your preference with python 3.9+ run the following

```bash
conda create --name YOUR_ENV_NAME python==3.9 
conda activate YOUR_ENV_NAME
pip install -r requirements.txt
```

```bash
python -m venv YOUR_ENV_NAME
source YOUR_ENV_NAME/bin/activate
pip install -r requirements.txt
```

# Usage

- Approach 1: traditional pypdf|pymuPDF parsers to extract text,images and tables. See `pymupdf_parse.py`

- Approach 2: Paddle Paddle PP-StructureV2: Layout detection + TableRec and the SLA head + PaddleOCRV4 to detecting text sections and tables. See `paddle_parse.py`

    See more details in the [documentation](https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppstructure/docs/quickstart_en.md)

- Approach 3: Object detector + PaddleOCRV4 using a zero-shot object detector (Grounding Dino) to get bounding boxes for text paragraphs, figures and tables. These bboxes are cropped and sent to the PaddleOCR engine for text extraction. See `objdet_paddle_parse.py`