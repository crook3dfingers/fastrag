from fastrag_client.errors import (
    AuthenticationError,
    ConnectionError,
    FastRAGError,
    NotFoundError,
    PayloadTooLargeError,
    ServerError,
    ValidationError,
)


def test_base_error_carries_status_and_body():
    err = FastRAGError("fail", status_code=500, body='{"error":"boom"}')
    assert str(err) == "fail"
    assert err.status_code == 500
    assert err.body == '{"error":"boom"}'


def test_base_error_defaults_none():
    err = FastRAGError("fail")
    assert err.status_code is None
    assert err.body is None


def test_subclasses_inherit_from_base():
    assert issubclass(AuthenticationError, FastRAGError)
    assert issubclass(NotFoundError, FastRAGError)
    assert issubclass(ValidationError, FastRAGError)
    assert issubclass(PayloadTooLargeError, FastRAGError)
    assert issubclass(ServerError, FastRAGError)
    assert issubclass(ConnectionError, FastRAGError)


def test_subclass_carries_status():
    err = AuthenticationError("denied", status_code=401, body="unauthorized")
    assert err.status_code == 401
    assert isinstance(err, FastRAGError)
