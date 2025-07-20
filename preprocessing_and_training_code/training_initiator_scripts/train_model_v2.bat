@echo off
python -m spacy train config.cfg ^
  --output BS-Model ^
  --paths.train Spacy_proper_split_data/bs_train_data_v2.spacy ^
  --paths.dev Spacy_proper_split_data/bs_dev_data_v2.spacy ^
  --gpu-id 0
pause
