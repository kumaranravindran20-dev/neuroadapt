from neuroadapt import AdaptiveModel


def test_adaptation_improves_prediction():
    model = AdaptiveModel(learning_rate=0.1)

    x = 1
    target = 2

    pred_before = model.predict(x)
    model.adapt(x, target)
    pred_after = model.predict(x)

    assert abs(target - pred_after) < abs(target - pred_before)
