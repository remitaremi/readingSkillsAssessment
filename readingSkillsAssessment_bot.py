# -*- coding: utf-8 -*-
from aiogram import Bot, Dispatcher, executor, types
import re

API_TOKEN = <token>
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

generated_phrases = []

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

DEVICE = torch.device("cpu")
model_name_or_path = <path>
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
model = GPT2LMHeadModel.from_pretrained(model_name_or_path).to(DEVICE)

@dp.message_handler(commands=['start'])
async def cmd_start(message: types.Message):
    kb = [
        [
            types.KeyboardButton(text="Простой"),
            types.KeyboardButton(text="Сложный")
        ],
    ]
    keyboard = types.ReplyKeyboardMarkup(
        keyboard=kb,
        resize_keyboard=True,
        input_field_placeholder="Выберите сложность текста"
    )
    await message.answer("Привет! Я могу генерировать тексты для оценки навыков чтения у детей. Выберите сложность текста.", reply_markup=keyboard)

@dp.message_handler()
async def echo(message: types.Message):

    text = "{\"label\": \"" + message.text + "\", \"text\": \""
    cut_len = len(text)
    max_len = 50
    while len(text) <= 1300:
        input_ids = tokenizer.encode(text, return_tensors="pt").to(DEVICE)
        model.eval()
        with torch.no_grad():
            out = model.generate(input_ids,
                                 do_sample=True,
                                 max_new_tokens=max_len,
                                 repetition_penalty=6.0,
                                 top_k=5,
                                 top_p=0.95,
                                 temperature=1.2,
                                 )

        generated_text = list(map(tokenizer.decode, out))[0]
        ind = generated_text.find('}')
        if ind and ind > 0:
            generated_text = generated_text[:ind - 1]
        generated_text_cut = generated_text[len(text):]
        phrase_regex = r"([^.!?]+[.!?])([^.!?]+[.!?])?"
        if re.match(phrase_regex, generated_text_cut):
           phrase = re.match(phrase_regex, generated_text_cut)
           if phrase.group(1):
               generated_text_cut = phrase.group(1)
               if phrase.group(2):
                   generated_text_cut += phrase.group(2)
        text += generated_text_cut
        print(text)
        if generated_text_cut == "":
            break
    bot_answer = text[cut_len:]
    bot_answer = re.sub(r"[A-z&]", "", bot_answer)
    bot_answer = re.sub(r"\s?[,;]{2,}\.?", ".", bot_answer)
    bot_answer = re.sub(r"\s{2,}", " ", bot_answer)
    await message.answer(bot_answer)

if __name__ == '__main__':
   executor.start_polling(dp, skip_updates=True)