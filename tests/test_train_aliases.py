from carsrl.train import apply_overrides, build_parser


def _cfg_from_args(*args: str):
    parser = build_parser()
    parsed = parser.parse_args(["--config", "configs/default.yaml", *args])
    return apply_overrides(parsed.config, parsed)


def test_random_phi_alias_uses_random_appraiser() -> None:
    cfg = _cfg_from_args("--algo", "ppo_random_phi")
    assert cfg.algo == "ppo_random_phi"
    assert cfg.cars.appraiser == "random"


def test_shuffled_phi_alias_uses_shuffled_qwen_appraiser() -> None:
    cfg = _cfg_from_args("--algo", "ppo_shuffled_phi")
    assert cfg.algo == "ppo_shuffled_phi"
    assert cfg.cars.appraiser == "shuffled_qwen"


def test_rnd_algo_is_configurable() -> None:
    cfg = _cfg_from_args("--algo", "ppo_rnd", "--intrinsic_coef", "0.07")
    assert cfg.algo == "ppo_rnd"
    assert cfg.intrinsic.coef == 0.07


def test_icm_and_ride_algos_are_configurable() -> None:
    icm_cfg = _cfg_from_args("--algo", "ppo_icm")
    ride_cfg = _cfg_from_args("--algo", "ppo_ride_cars", "--cars_appraiser", "mock")
    assert icm_cfg.algo == "ppo_icm"
    assert ride_cfg.algo == "ppo_ride_cars"
    assert ride_cfg.cars.appraiser == "mock"


def test_no_confidence_alias_disables_confidence() -> None:
    cfg = _cfg_from_args("--algo", "ppo_cars_no_confidence")
    assert cfg.cars.use_confidence is False


def test_direct_alias_enables_direct_reward() -> None:
    cfg = _cfg_from_args("--algo", "ppo_cars_direct")
    assert cfg.cars.direct_reward is True


def test_history_and_skip_flags_are_configurable() -> None:
    history_cfg = _cfg_from_args("--algo", "ppo_cars", "--cars_history")
    shape_on_skip_cfg = _cfg_from_args("--algo", "ppo_cars", "--cars_shape_on_skip")

    assert history_cfg.cars.include_history is True
    assert shape_on_skip_cfg.cars.neutral_on_skip is False
