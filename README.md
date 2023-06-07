# uit_subjects_recommendation_system
conda install -c conda-forge tqdm
conda install -c conda-forge pandas
conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install -c anaconda scikit-learn

git clone --single-branch --branch fast_tokenizers_BARTpho_PhoBERT_BERTweet https://github.com/datquocnguyen/transformers.git
cd transformers
pip3 install -e .