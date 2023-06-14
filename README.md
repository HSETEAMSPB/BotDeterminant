# BotDeterminant

Telegram bot, on the input of which is a picture and voice message. 
The bot turns speech into a sentence and makes it analyzed, thereby highlighting the main word (the subject in question). 
In the response, the original image with the highlighted item is sent.

## What we used
- for ASR module we've implemented rnn trancducer like model, based on [article](https://arxiv.org/abs/2005.03191)
- for text processing (parsing) we've used the nltk & spacy libraries
- for CV module we've implemented [Yolo](https://arxiv.org/abs/1804.02767) like model

## Opportunities for asr
fro ASR You have the following options right into your bot:
 - noise reduction - you can connect the removal of background noise when recognizing audio
 - vad hardness - you can set a threshold for voice activity detector

## Bot setup
you need to open the bot.ipynb in [Google Colab](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjkyIT6vsL_AhWDyIsKHSMTBHYQFnoECA4QAQ&url=https%3A%2F%2Fresearch.google.com%2Fcolaboratory%2F&usg=AOvVaw38J01zt_Dlb6pQ1fe6FGrI) and make cells in sequence

## Configuration
you need to put the token from your bot directly into bot.ipynb
also you have a lot of configurations for modules (config.py files), so far they are installed by default

## Launch example
### telegram bot screenshots
<p float="left">
  <img src="/launch_files/first.jpg" width="250" />
  <img src="/launch_files/2nd.jpg" width="250" />
</p>

### Logs
Voice message:

[![voice](launch_files/2nd.png)](launch_files/launch_files_voice.mp3)





In/Out images:
<p float="left">
  <img src="/launch_files/musk.jpg" width="250" />
  <img src="/launch_files/detected_musk.jpg" width="250" />
</p>
