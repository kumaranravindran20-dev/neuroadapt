def test_adaptive_learning():
    from neuroadapt import AdaptiveModel

    model = AdaptiveModel()
    before = model.predict(1.0)
    model.adapt(1.0, 2.0)
    after = model.predict(1.0)

    assert after != before
    print("AdaptiveModel learning test passed.")