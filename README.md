# OCR Tools for parsing PDF Files

This repo contains tools for parsing pdf files and its content using open source frameworks and models.


- Approach 1: traditional pypdf|pymuPDF parsers to extract text,images and tables. See `pymupdf_parse.py`

- Approach 2: Paddle Paddle PP-StructureV2: Layout detection + TableRec and the SLA head + PaddleOCRV4 to detecting text sections and tables. See `paddle_parse.py`

    See more details in the [documentation](https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppstructure/docs/quickstart_en.md)

- Approach 3: Object detector + PaddleOCRV4 using a zero-shot object detector (Grounding Dino) to get bounding boxes for text paragraphs, figures and tables. These bboxes are cropped and sent to the PaddleOCR engine for text extraction. See `objdet_paddle_parse.py`