# mmpose_ABLH_predict

### 安裝mmpose
  MMPose works on Linux, Windows and macOS. It requires Python 3.7+, CUDA 9.2+ and PyTorch 1.6+.

  首先安裝Anaconda，完成後執行下列指令
	conda create -n openmmlab python=3.8  cudatoolkit=11.3 -y
	conda activate openmmlab

  接著安裝pytorch，建議從pytorch官方網站選擇和顯卡cuda版本相容的pytorch版本安裝
  注意torch版本號後面要有+cu，如: pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

	pip install openmim
	mim install mmcv==1.7
	cd mmpose
	pip install -e .


### 訓練
  訓練時需要先進到mmpose/目錄下，並用python執行tools/train.py，指令如下

	python tools/train.py ${MMPOSE_CONFIG_FILE} [--resume-from ${MMPOSE_CHECKPOINT_FILE}] [--work-dir ${CHECKPOINT_DIR}]

  訓練出的權重檔預設儲存在work_dirs/
  --resume-from 從指定權重檔重新開始訓練
  --work-dir 指定儲存權重檔的位置
  

### 預測
  預測時需要先進到mmpose/目錄下，並用python執行top_down_img_predict.py
	python top_down_img_predict.py \
		${MMPOSE_CONFIG_FILE} ${MMPOSE_CHECKPOINT_FILE} \
		--csv-root ${CSV_ROOT} \
		--img-root ${IMG_ROOT} --json-file ${JSON_FILE} \
		--out-root ${OUTPUT_DIR} \
		[--show --device ${GPU_ID}] \
		[--kpt-thr ${KPT_SCORE_THR}]

  preprocess.py、top_down_img_predict.py和mmpose/需放在同一資料夾。

  --csv-root 輸入.csv的位置，{CSV_ROOT}/日期/level1/.csv。

  --img-root 用於預測的圖片位置。
  --json-file 和用於預測的圖片成對的annotatoins的位置。

  --out-root 儲存輸出結果的位置，輸出結果包含: 預測結果的.csv、為了預測產生的annotaion和存放產生的圖片的資料夾。

  如果有輸入--csv-root，程式會從.csv檔產生預測需要的圖片和annotations；如果輸入--img-root，程式會從已有的圖片進行預測，且這時需要再透過--json-file提供annotation的位置。
