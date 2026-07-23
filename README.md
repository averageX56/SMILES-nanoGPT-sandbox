# SMILES-nanoGPT-sandbox

Репозиторий представляет собой песочницу на базе трансформера для проведения экспериментов над всей архитектурой: и над конфигами обучения, и над самой моделью.

---

# 1. Как использовать

## Клонирование репозитория

```bash
git clone https://github.com/averageX56/SMILES-nanoGPT-sandbox.git
cd SMILES-nanoGPT-sandbox
```

## Установка зависимостей

```bash
pip install -r requirements.txt
```

## Запуск обучения

Для запуска обучения необходимо указать файл конфигурации:

```bash
python train.py config/small.py
```

По умолчанию используется архитектура `base` из папки `models/`. Любой параметр можно переопределить прямо из командной строки:

```bash
python train.py config/small.py --batch_size=64 --max_iters=2000
```

---

## Добавление собственного конфига (обучения)

Новый конфиг можно создать, скопировав любой из существующих файлов из директории `config/`:

```text
config/
├── small.py
├── medium.py
├── large.py
└── my_config.py
```

После создания собственного конфига модель запускается аналогичным образом:

```bash
python train.py config/my_config.py
```

---

## Выбор архитектуры

Помимо конфига обучения, теперь можно менять и саму **модель**. Архитектуры лежат в отдельной папке `models/`, и нужная выбирается флагом `--model`:

```text
models/
├── __init__.py     # реестр моделей + загрузчик get_model()
├── base.py         # архитектура по умолчанию (--model=base)
└── my_model.py     # ваша собственная архитектура
```

```bash
# обучение с архитектурой по умолчанию (можно не указывать)
python train.py config/small.py --model=base

# обучение со своей архитектурой из models/my_model.py
python train.py config/small.py --model=my_model
```

Значение флага `--model` — это просто имя файла в папке `models/` без расширения `.py`.

---

## Структура проекта

```text
.
├── train.py          # обучение модели
├── sample.py         # генерация из одного промпта
├── inference.py      # инференс на выборке (файл с промптами)
├── models/           # архитектуры моделей (выбираются флагом --model)
│   ├── __init__.py
│   └── base.py
├── generator/        # генераторы задач (выбираются флагом --dataset)
│   ├── __init__.py   # реестр генераторов + get_generator()
│   ├── base.py       # абстрактный Generator, collate с маскированием ответа
│   └── kv_retrieval.py
├── config/           # конфигурации обучения
├── eval/             # оценка чекпоинтов + экспорт CSV (запуск: python eval/<script>.py, из корня репо)
│   ├── __init__.py
│   ├── task_eval_utils.py              # общие хелперы: загрузка чекпоинта, генератор (любая задача), accuracy
│   ├── task_inference.py               # тестовый инференс с чекпоинта на любой задаче (флаг --dataset)
│   ├── export_training_dynamics_csv.py  # CSV динамики обучения из --log_file
│   ├── export_checkpoint_scores_csv.py  # CSV скоров нескольких чекпоинтов (любая задача)
│   └── export_pdf_metrics_csv.py         # CSV метрик из Smiles_Transormers.pdf
└── out-*             # сохранённые модели и логи обучения
```

---

# 2. Преднастроенные конфиги

В репозитории присутствуют три готовые конфигурации моделей.

| Конфиг | Размер модели | Архитектура |
|-------|-------|-------|
| `small.py` | ~2.2M параметров | 4 слоя, 4 головы, 128 embedding |
| `medium.py` | ~7.5M параметров | 7 слоёв, 6 голов, 240 embedding |
| `large.py` | ~30.8M параметров | 8 слоёв, 8 голов, 512 embedding |

# 3. Данные: онлайн-генерация

Данные для обучения и валидации **не хранятся на диске** — они стримятся на лету из
генератора задачи. Никаких `train.bin` / `val.bin` / `meta.pkl` больше не требуется.

Задача выбирается флагом `--dataset` (имя генератора в реестре `generator/`), например:

```bash
python train.py config/small.py --dataset=kv
```

На каждом шаге `train.py`:
- тренировочный батч сэмплируется заново (`gen.sample_train(batch_size)`);
- валидация — фиксированный отложенный набор (`gen.generate_val(n_val)`), дедуплицированный
  относительно train по хешу промпта;
- `gen.collate(items, block_size)` упаковывает примеры `prompt → answer` в тензоры с
  **маскированием функции потерь на ответе** (все позиции кроме токенов ответа = `-1`,
  игнорируются в `F.cross_entropy(ignore_index=-1)`);
- `vocab_size` берётся напрямую из генератора (`gen.vocab_size`).

Параметры задачи задаются через `gen_params` в конфиге (мёржатся поверх умолчаний реестра)
и записываются в чекпойнт (`ckpt['config']['gen_params']`) для воспроизводимости. Размер
отложенной валидации — `n_val`. Во время оценки печатается и loss, и **exact-match точность
ответа** (`--eval_accuracy`).

### Контракт генератора

Чтобы добавить новую задачу, создайте файл в `generator/` с подклассом `Generator`
(см. `generator/base.py`) и одну запись в реестре `GENERATORS` в `generator/__init__.py`.
Подкласс должен реализовать:

- `_sample_one() -> DatasetItem(prompt, answer)` — один пример «промпт → ответ»;
- свойство `vocab_size` — число используемых токенов;
- атрибут класса `PAD_ID` (по умолчанию `0`).

Всё остальное базовый класс предоставляет бесплатно: `sample_train` / `generate_val`
(с дедупликацией по хешу промпта) и общий `collate` с маскированием ответа. Имя в конфиге
не обязано совпадать с именем файла, и несколько связанных задач могут жить в одном модуле.

---

# 4. Инференс

## 4.1. Генерация из одного промпта (`sample.py`)

Для генерации из конкретного промпта можно воспользоваться параметром `--start`:

```bash
python sample.py --out_dir=out-small --start="CCO"
```

Также можно изменить количество генерируемых токенов и число сэмплов:

```bash
python sample.py \
    --out_dir=out-small \
    --start="CCO" \
    --num_samples=5 \
    --max_new_tokens=100
```

## 4.2. Инференс на выборке (`inference.py`)

Файл `inference.py` прогоняет модель по текстовому файлу, где каждая строка это отдельный промпт. Для каждой строки генерируется одна или несколько достроек.


Пример файла `prompts.txt`:

```text
CCO
c1ccccc1
CC(=O)

```

### Запуск

```bash
python inference.py \
    --out_dir=out-small \
    --input_file=prompts.txt \
    --output_file=generations.csv \
    --num_samples=5 \
    --max_new_tokens=100 \
    --temperature=0.8
```

Полезные флаги:

| Флаг | Значение по умолчанию | Описание |
|------|-----------------------|----------|
| `--input_file` | — (обязательный) | файл выборки, по одному промпту на строку |
| `--output_file` | `''` | куда сохранить CSV с результатами (если пусто — только вывод в консоль) |
| `--num_samples` | `1` | сколько достроек генерировать на каждый промпт |
| `--max_new_tokens` | `100` | сколько токенов дописывать после промпта |
| `--temperature` | `0.8` | температура  |
| `--top_k` | `200` | ограничение на топ-k наиболее вероятных токенов |
| `--stop_on_newline` | `True` | обрезать генерацию по первому переводу строки (одна молекула на строку) |

Если задан `--output_file`, на выходе будет CSV со столбцами:

```text
prompt_index, prompt, sample_index, generation
```

---

# 5. Тестовый инференс на чекпоинте (`eval/task_inference.py`)

```bash
python eval/task_inference.py --out_dir=out-small
```

| Флаг | Значение по умолчанию | Описание |
|------|-----------------------|----------|
| `--out_dir` | `'out-small'` | директория с чекпоинтом (`ckpt.pt`) |
| `--dataset` | `''` | задача/генератор (как в `train.py`); пусто = та, что в чекпоинте |
| `--gen_param_overrides` | `[]` | список словарей, каждый мёржится поверх `gen_params` чекпоинта — одна строка вывода на словарь; пусто = только обученная конфигурация |
| `--n_eval` | `500` | сколько свежих примеров на строку |
| `--summary_csv` / `--examples_csv` | `''` | куда сохранить сводный CSV / CSV с отдельными примерами |

```bash
# KV-retrieval — OOD-развёртка по n_pairs
python eval/task_inference.py --out_dir=out-small \
    --gen_param_overrides="[{'n_pairs':96},{'n_pairs':192},{'n_pairs':384}]" \
    --summary_csv=kv_inference_summary.csv \
    --examples_csv=kv_inference_examples.csv

# другая задача, тот же скрипт — addition, OOD по числу цифр
python eval/task_inference.py --out_dir=out-addition --dataset=addition \
    --gen_param_overrides="[{'n_digits':10},{'n_digits':20}]"

```

## 5.2. Динамика обучения (`eval/export_training_dynamics_csv.py`)

`train.py` теперь поддерживает необязательный `--log_file=path.jsonl`:
по одной JSON-строке на каждый `eval_interval` (iter, train/val loss,
train/val acc, lr, mfu; по умолчанию `''` — выключено, поведение не меняется).
Экспортёр читает этот JSONL — или обычный текстовый лог, перехваченный через
`tee` — и пишет по отдельному двухколоночному CSV (`iter, значение`) на
каждую метрику:

```bash
python train.py config/small.py --log_file=out-small/train_log.jsonl
python eval/export_training_dynamics_csv.py --log_file=out-small/train_log.jsonl \
    --csv_out_dir=out-small/dynamics_csv
```

# 6. Ноутбуки

Также в репозитории лежат ноутбуки из оригинального репозитория nanoGPT: `transformer_sizing.ipynb` и `scaling_laws.ipynb` — их можно использовать для оценки размеров модели и scaling laws.
