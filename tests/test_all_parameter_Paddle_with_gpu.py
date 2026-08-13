import os

# ---------------------------------------------------------------------------
# Biến môi trường khuyến nghị cho PaddleOCR (thiết lập sớm trước mọi import)
# ---------------------------------------------------------------------------
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

######################################################### TextDetection
from paddleocr import TextDetection
modeltd = TextDetection(model_name="PP-OCRv5_mobile_det", #Meaning:Model name. Description: If set to None, PP-OCRv5_server_det will be used.
                        model_dir=None, #Meaning:Model storage path.
                        device="gpu", #Meaning:Device for inference. Description: For example:"cpu", "gpu", "npu", "gpu:0", "gpu:0,1". If multiple devices are specified, parallel inference will be performed. By default, GPU 0 is used if available; otherwise, CPU is used.
                        enable_hpi=False, #Meaning:Whether to enable high-performance inference.
                        use_tensorrt=False, #Meaning:Whether to use the Paddle Inference TensorRT subgraph engine. Description: If the model does not support acceleration through TensorRT, setting this flag will not enable acceleration. For Paddle with CUDA version 11.8, the compatible TensorRT version is 8.x (x>=6), and it is recommended to install TensorRT 8.6.1.6.
                        precision="fp32", #Meaning:Computation precision when using the Paddle Inference TensorRT subgraph engine. Description: Options: "fp32", "fp16".
                        enable_mkldnn=True, #Meaning:Whether to enable MKL-DNN acceleration for inference. Description: If MKL-DNN is unavailable or the model does not support it, acceleration will not be used even if this flag is set.
                        mkldnn_cache_capacity=16, #Meaning:MKL-DNN cache capacity.
                        cpu_threads=16, #Meaning:Number of threads to use for inference on CPUs.
                        limit_side_len=None, #Meaning:Limit on the side length of the input image for detection. Description: int specifies the value. If set to None, the model's default configuration will be used.
                        limit_type=None, #Meaning:Type of image side length limitation. Description: "min" ensures the shortest side of the image is no less than det_limit_side_len; "max" ensures the longest side is no greater than limit_side_len. If set to None, the model's default configuration will be used.
                        thresh=None, #Meaning:Pixel score threshold. Pixels in the output probability map with scores greater than this threshold are considered text pixels. Description: If set to None, the model's default configuration will be used.
                        box_thresh=None, #Meaning:If the average score of all pixels inside the bounding box is greater than this threshold, the result is considered a text region. Description: If set to None, the model's default configuration will be used.
                        unclip_ratio=None, #Meaning:Expansion ratio for the Vatti clipping algorithm, used to expand the text region. Description: If set to None, the model's default configuration will be used.
                        input_shape=None) #Meaning:Input image size for the model in the format (C, H, W).
outputtd = modeltd.predict(input="frame_001874_84960ms.jpg", #Meaning:Input data to be predicted. Required. Description: Supports multiple input types: Python variable: e.g., numpy.ndarray representing image data str: Local image file or PDF file path: /root/data/img.jpg; URL: Image or PDF file network URL: Example; Directory: Should contain images for prediction, e.g., /root/data/ (currently, PDF files in directories are not supported, PDF files need to be specified by file path) list: List elements should be of the above types, e.g., [numpy.ndarray, numpy.ndarray], ["/root/data/img1.jpg", "/root/data/img2.jpg"], ["/root/data1", "/root/data2"]
                         batch_size=1, #Meaning:Batch size. Description:Positive integer.
                         limit_side_len=None, #Meaning:Same meaning as the instantiation parameters. Description: If set to None, the instantiation value is used; otherwise, this parameter takes precedence.
                         limit_type=None, #Meaning:Same meaning as the instantiation parameters. Description: If set to None, the instantiation value is used; otherwise, this parameter takes precedence.
                         thresh=None, #Meaning:Same meaning as the instantiation parameters. Description: If set to None, the instantiation value is used; otherwise, this parameter takes precedence.
                         box_thresh=None, #Meaning:Same meaning as the instantiation parameters. Description: If set to None, the instantiation value is used; otherwise, this parameter takes precedence.
                         unclip_ratio=None) #Meaning:Same meaning as the instantiation parameters. Description: If set to None, the instantiation value is used; otherwise, this parameter takes precedence.
for restd in outputtd:
    restd.print() #Print results to terminal
    restd.save_to_img(save_path="./output/") #Save results as image
    restd.save_to_json(save_path="./output/res.json") #Save results as JSON file

#The output will be:{'res': {'input_path': 'general_ocr_001.png', 'page_index': None, 'dt_polys': array([[[ 75, 549], ..., [ 77, 586]], ..., [[ 31, 406], ..., [ 34, 455]]], dtype=int16), 'dt_scores': [0.873949039891189, 0.8948166013613552, 0.8842595305917041, 0.876953790920377]}}
#Output parameter meanings:
#input_path：Path of the input image.
#page_index：If the input is a PDF, this indicates the current page number; otherwise, it is None
#dt_polys：Predicted text detection boxes, where each box contains four vertices (x, y coordinates).
#dt_scores：Confidence scores of the predicted text detection boxes.

######################################################### TextRecognition
from paddleocr import TextRecognition
modeltr = TextRecognition(model_name="PP-OCRv5_mobile_rec", #Description: If set to None, PP-OCRv5_server_rec is used.
                          model_dir=None, #Meaning:Model storage path.
                          device="gpu", #Meaning: Device for inference. Description: Examples: "cpu", "gpu", "npu", "gpu:0", "gpu:0,1". If multiple devices are specified, inference will be performed in parallel. By default, GPU 0 is used; if unavailable, CPU is used.
                          enable_hpi=False, #Meaning: Whether to enable high performance inference.
                          use_tensorrt=False, #Meaning: Whether to enable the TensorRT subgraph engine of Paddle Inference. Description: For Paddle with CUDA 11.8, the compatible TensorRT version is 8.x (x>=6), recommended 8.6.1.6.
                          precision="fp32", #Meaning:Precision for TensorRT when using the Paddle Inference TensorRT subgraph engine. Description: Options: fp32, fp16.
                          enable_mkldnn=True, #Meaning: Whether to enable MKL-DNN acceleration for inference. Description: If MKL-DNN is unavailable or the model does not support it, acceleration will not be used even if this flag is set.
                          mkldnn_cache_capacity=16, #Meaning: MKL-DNN cache capacity.
                          cpu_threads=16, #Meaning:Number of threads to use for inference on CPUs.
                          input_shape=None) #Meaning:Input image size for the model in the format (C, H, W).
#Call the predict() method of the text recognition model for inference. This method returns a list of results. In addition, this module also provides the predict_iter() method. The two methods are completely consistent in terms of parameter acceptance and result return. The difference is that predict_iter() returns a generator, which can process and obtain prediction results step by step. It is suitable for scenarios where large datasets need to be processed or memory savings are desired. You can choose either of these two methods according to your actual needs. The parameters of the predict() method include input and batch_size, with specific descriptions as follows:
outputtr = modeltr.predict(input="frame_001874_84960ms.jpg", #Meaning:Data to be predicted, supporting multiple input types, required. Description: Python Var: Image data represented by numpy.ndarray str: Local path of image file or PDF file: /root/data/img.jpg; URL link: Network URL of image file or PDF file: Example; Local directory: The directory should contain the images to be predicted, such as /root/data/ (currently, prediction of PDF files in the directory is not supported, PDF files need to be specified to a specific file path) list: The elements of the list should be data of the above types, such as [numpy.ndarray, numpy.ndarray], ["/root/data/img1.jpg", "/root/data/img2.jpg"], ["/root/data1", "/root/data2"]
                           batch_size=1) #Batch size, can be set to any positive integer.
for restr in outputtr:
    restr.print() #Print the result to the terminal
    restr.save_to_img(save_path="./output/") #Save the result as a file in image format
    restr.save_to_json(save_path="./output/res.json") #Save the result as a file in json format

#After running, the result is as follows: {'res': {'input_path': 'general_ocr_rec_001.png', 'page_index': None, 'rec_text': '绿洲仕格维花园公寓', 'rec_score': 0.9823867082595825}}
#The meanings of the parameters in the result are as follows: - input_path: The path of the input text line image to be predicted - page_index: If the input is a PDF file, it indicates which page of the PDF the current text line is from; otherwise, it is None - rec_text: The predicted text of the text line image - rec_score: The confidence score of the predicted text for the text line image

######################################################### PaddleOCR
from paddleocr import PaddleOCR  
ocr = PaddleOCR(device="gpu", #Meaning:Device for inference. Description: Supports specifying a specific card number: CPU: e.g., cpu for CPU inference; GPU: e.g., gpu:0 for inference on the 1st GPU; NPU: e.g., npu:0 for inference on the 1st NPU; XPU: e.g., xpu:0 for inference on the 1st XPU; MLU: e.g., mlu:0 for inference on the 1st MLU; DCU: e.g., dcu:0 for inference on the 1st DCU; MetaX GPU: e.g., metax_gpu:0 for inference on the 1st MetaX GPU; Iluvatar GPU: e.g., iluvatar_gpu:0 for inference on the 1st Iluvatar GPU; None: If set to None, the pipeline initialized value for this parameter will be used. During initialization, the local GPU device 0 will be preferred; if unavailable, the CPU device will be used.
                text_detection_model_name="PP-OCRv5_mobile_det", #Meaning:Name of the text detection model. Description: If set to None, the pipeline's default model will be used.
                text_detection_model_dir=None, #Meaning:Directory path of the text detection model. Description: If set to None, the official model will be downloaded.
                text_recognition_model_name="PP-OCRv5_mobile_rec", #Meaning:Name of the text recognition model. Description: If set to None, the pipeline's default model will be used.
                text_recognition_model_dir=None, #Meaning:Directory path of the text recognition model. Description: If set to None, the official model will be downloaded.
                text_recognition_batch_size=None, #Meaning:Batch size for the text recognition model. Description: If set to None, the default batch size will be 1
                use_doc_orientation_classify=False, # Meaning:Whether to load and use the document orientation classification module. Description: If set to None, the pipeline's initialized value for this parameter (defaults to True) will be used.
                use_doc_unwarping=False, # Meaning:Whether to load and use the text image unwarping module. Description: If set to None, the pipeline's initialized value for this parameter (defaults to True) will be used.
                use_textline_orientation=False, #Meaning:Whether to load and use the text line orientation module. Description: If set to None, the pipeline's initialized value for this parameter (defaults to True) will be used.
                doc_orientation_classify_model_name=None, #Meaning:Name of the document orientation classification model.Description: If set to None, the pipeline's default model will be used.
                doc_orientation_classify_model_dir=None, #Meaning:Directory path of the document orientation classification model. Description: If set to None, the official model will be downloaded.
                doc_unwarping_model_name=None, #Meaning:Name of the text image unwarping model. Description: If set to None, the pipeline's default model will be used.
                doc_unwarping_model_dir=None, #Meaning:Directory path of the text image unwarping model. Description: If set to None, the official model will be downloaded.
                textline_orientation_model_name=None, #Meaning:Name of the text line orientation model.Description: If set to None, the pipeline's default model will be used.
                textline_orientation_model_dir=None, #Meaning:Directory path of the text line orientation model. Description: If set to None, the official model will be downloaded.
                textline_orientation_batch_size=None, #Meaning:Batch size for the text line orientation model. Description: If set to None, the default batch size will be 1.
                text_det_limit_side_len=None, # Meaning:Image side length limitation for text detection. Description: int: Any integer greater than 0; None: If set to None, the pipeline's initialized value for this parameter (defaults to 64) will be used.
                text_det_limit_type=None, #Meaning:Type of side length limit for text detection. Description: str: Supports min and max, where min means ensuring the shortest side of the image is not smaller than det_limit_side_len, and max means ensuring the longest side of the image is not larger than limit_side_len; None: If set to None, the pipeline's initialized value for this parameter (defaults to min) will be used.
                text_det_thresh=None, #Meaning:Pixel threshold for text detection. Pixels with scores higher than this threshold in the output probability map will be considered text pixels. Description: float: Any floating-point number greater than 0; None: If set to None, the pipeline's initialized value for this parameter (defaults to 0.3) will be used.
                text_det_box_thresh=None, #Meaning:Box threshold for text detection. A detection result will be considered a text region if the average score of all pixels within the bounding box is higher than this threshold. Description: float: Any floating-point number greater than 0; None: If set to None, the pipeline's initialized value for this parameter (defaults to 0.6) will be used.
                text_det_unclip_ratio=None, #Meaning:Dilation coefficient for text detection. This method is used to dilate the text region, and the larger this value, the larger the dilated area. Description: float: Any floating-point number greater than 0; None: If set to None, the pipeline's initialized value for this parameter (defaults to 2.0) will be used.
                text_det_input_shape=None, #Meaning:Input shape for text detection.
                text_rec_score_thresh=None, #Meaning:Recognition score threshold for text. Text results with scores higher than this threshold will be retained. Description: float: Any floating-point number greater than 0; None: If set to None, the pipeline's initialized value for this parameter (defaults to 0.0, i.e., no threshold) will be used.
                text_rec_input_shape=None, #Meaning:Input shape for text recognition.
                lang=None, #Meaning:OCR model language to use. Description: The table in the appendix lists all the supported languages.
                ocr_version=None, #Meaning:Version of OCR models. Description: PP-OCRv5: Use PP-OCRv5 series models; PP-OCRv4: Use PP-OCRv4 series models; PP-OCRv3: Use PP-OCRv3 series models. Please note that not every ocr_version supports all lang options. Please refer to the correspondence table in the appendix for details.
                enable_hpi=False, #Meaning:Whether to enable high-performance inference.
                use_tensorrt=False, #Meaning:Whether to use the Paddle Inference TensorRT subgraph engine. If the model does not support acceleration through TensorRT, setting this flag will not enable acceleration. Description: For Paddle with CUDA version 11.8, the compatible TensorRT version is 8.x (x>=6), and it is recommended to install TensorRT 8.6.1.6.
                precision="fp32", #Meaning:Computational precision, such as fp32, fp16.
                enable_mkldnn=True, #Meaning:Whether to enable MKL-DNN acceleration for inference.Description: If MKL-DNN is unavailable or the model does not support it, acceleration will not be used even if this flag is set.
                mkldnn_cache_capacity=16, #Meaning:MKL-DNN cache capacity.
                cpu_threads=16, #Meaning:Number of threads used for CPU inference.
                paddlex_config=None #Meaning:Path to the PaddleX pipeline configuration file.
)
result = ocr.predict(input="./frame_001876_85120ms.jpg", #Meaning:Data to be predicted, supporting multiple input types, required. Description: Python Var: Image data represented by numpy.ndarray; str: Local path of an image file or PDF file: /root/data/img.jpg; URL link, such as the network URL of an image file or PDF file: example; local directory, which needs to contain images to be predicted, such as the local path: /root/data/ (currently, predicting PDF files in the directory is not supported; PDF files need to specify the specific file path); list: List elements must be of the above types, such as [numpy.ndarray, numpy.ndarray], ["/root/data/img1.jpg", "/root/data/img2.jpg"], ["/root/data1", "/root/data2"].
                     use_doc_orientation_classify=None, #Meaning:Whether to use the document orientation classification module during inference.
                     use_doc_unwarping=None, #Meaning:Whether to use the text image unwarping module during inference.
                     use_textline_orientation=None, #Meaning:Whether to use the text line orientation classification module during inference.
                     text_det_limit_side_len=None, #Meaning:Same meaning as the instantiation parameters. Description: If set to None, the instantiation value is used; otherwise, this parameter takes precedence.
                     text_det_limit_type=None, #Meaning:Same meaning as the instantiation parameters. Description: If set to None, the instantiation value is used; otherwise, this parameter takes precedence.
                     text_det_thresh=None, #Meaning:Same meaning as the instantiation parameters. Description: If set to None, the instantiation value is used; otherwise, this parameter takes precedence.
                     text_det_box_thresh=None, #Meaning:Same meaning as the instantiation parameters. Description: If set to None, the instantiation value is used; otherwise, this parameter takes precedence.
                     text_det_unclip_ratio=None, #Meaning:Same meaning as the instantiation parameters. Description: If set to None, the instantiation value is used; otherwise, this parameter takes precedence.
                     text_rec_score_thresh=None #Meaning:Same meaning as the instantiation parameters. Description: If set to None, the instantiation value is used; otherwise, this parameter takes precedence.
) #Invoke the predict() method of the OCR pipeline object for inference prediction, which returns a results list. Additionally, the pipeline provides the predict_iter() method. Both methods are completely consistent in parameter acceptance and result return, except that predict_iter() returns a generator, which can process and obtain prediction results incrementally, suitable for handling large datasets or scenarios where memory saving is desired. You can choose to use either of these two methods according to actual needs. The following are the parameters and descriptions of the predict() method:
for res in result:  
    res.print()  
    res.save_to_img("output")  
    res.save_to_json("output")

#Results are printed to the terminal:
#{'res': {'input_path': './general_ocr_002.png', 'page_index': None, 'model_settings': {'use_doc_preprocessor': True, 'use_textline_orientation': False}, 'doc_preprocessor_res': {'input_path': None, 'page_index': None, 'model_settings': {'use_doc_orientation_classify': False, 'use_doc_unwarping': False}, 'angle': -1}, 'dt_polys': array([[[  3,  10], ..., [  4,  30]], ..., [[ 99, 456], ..., [ 99, 479]]], dtype=int16), 'text_det_params': {'limit_side_len': 736, 'limit_type': 'min', 'thresh': 0.3, 'max_side_limit': 4000, 'box_thresh': 0.6, 'unclip_ratio': 1.5}, 'text_type': 'general', 'textline_orientation_angles': array([-1, ..., -1]), 'text_rec_score_thresh': 0.0, 'rec_texts': ['www.997700', '', 'Cm', '登机牌', 'BOARDING', 'PASS', 'CLASS', '序号SERIAL NO.', '座位号', 'SEAT NO.', '航班FLIGHT', '日期DATE', '舱位', '', 'W', '035', '12F', 'MU2379', '03DEc', '始发地', 'FROM', '登机口', 'GATE', '登机时间BDT', '目的地TO', '福州', 'TAIYUAN', 'G11', 'FUZHOU', '身份识别IDNO.', '姓名NAME', 'ZHANGQIWEI', '票号TKT NO.', '张祺伟', '票价FARE', 'ETKT7813699238489/1', '登机口于起飞前10分钟关闭 GATESCL0SE10MINUTESBEFOREDEPARTURETIME'], 'rec_scores': array([0.67634439, ..., 0.97416091]), 'rec_polys': array([[[  3,  10], ..., [  4,  30]], ..., [[ 99, 456], ..., [ 99, 479]]], dtype=int16), 'rec_boxes': array([[  3, ...,  30], ..., [ 99, ..., 479]], dtype=int16)}}