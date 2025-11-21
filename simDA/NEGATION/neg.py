from typing import List
from typing import Tuple

import spacy
from checklist.expect import Expect
from checklist.perturb import Perturb
from checklist.pred_wrapper import PredictorWrapper
from checklist.test_types import MFT
from model import SentimentModel


def run_negation_test(model: SentimentModel, sents: List[str]) -> MFT:
    # Загружаем модель spacy для обработки текста
    nlp = spacy.load("en_core_web_sm")

    # Обертываем модель для checklist
    def predict_fn(texts):
        probs = model.predict_proba(texts)
        preds = probs.argmax(axis=1)
        return preds, probs

    wrapped_model = PredictorWrapper.wrap_predict(predict_fn)

    # Создаем тестовые случаи: оригинальные предложения и их версии с отрицанием
    testcases = []
    for sent in sents:
        # Добавляем отрицание к предложению
        perturbed = Perturb.add_negation(sent, nlp)
        testcases.append([sent, perturbed])

    # Определяем функцию response для проверки тестов
    def response(xs, preds, confs, labels=None, meta=None):
        results = []
        # Обрабатываем пары предложений (оригинал + с отрицанием)
        for i in range(0, len(xs), 2):
            if i + 1 < len(xs):
                # Берем уверенность позитивного класса для обоих предложений
                conf_original = confs[i][2]  # индекс 2 - позитивный класс
                conf_negated = confs[i + 1][2]

                # Проверяем разницу в уверенности
                diff = abs(conf_original - conf_negated)
                if diff > 0.3:
                    # Тест пройден - большая разница в уверенности
                    results.extend([True, True])
                else:
                    # Тест не пройден - маленькая разница
                    results.extend([False, False])
        return results

    # Создаем и запускаем тест
    test = MFT(
        data=testcases, expect=Expect.single(response), wrapped_model=wrapped_model
    )
    test.run()

    return test


if __name__ == "__main__":
    model = SentimentModel()
    sents = [
        "The delivery was swift and on time.",
        "I wasn't disappointed with the service.",
        "The food arrived cold and unappetizing.",
        "Their app is quite user-friendly and intuitive.",
        "I didn't find their selection lacking.",
        "The delivery person was rude and impatient.",
        "They always have great deals and offers.",
        "I haven't had any bad experiences yet.",
        "I was amazed by the quick response to my complaint.",
        "Their tracking system isn't always accurate.",
    ]

    test = run_negation_test(model, sents)

    def format_example(x, pred, conf, label=None, meta=None):
        return f"{x} (pos. class conf.: {conf[2]:.2f})"

    print(test.summary(n=5, format_example_fn=format_example))
