from main import main


def test_main(capsys) -> None:
    main()
    captured = capsys.readouterr()
    assert "42 Template Python" in captured.out
