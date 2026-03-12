"""
Unit tests for the SQL validation layer.
"""

import pytest
from app.validators.sql_validator import SQLValidator


@pytest.fixture
def validator():
    return SQLValidator()


class TestSQLValidator:

    def test_valid_aggregation_query(self, validator):
        sql = "SELECT categoryname, COUNT(ticketnumber) AS cnt FROM tickets GROUP BY categoryname"
        result = validator.validate(sql)
        assert result.is_valid is True
        assert result.errors == []

    def test_rejects_empty(self, validator):
        result = validator.validate("")
        assert result.is_valid is False
        assert any("Empty" in e for e in result.errors)

    def test_rejects_insert(self, validator):
        sql = "INSERT INTO tickets (title) VALUES ('hack')"
        result = validator.validate(sql)
        assert result.is_valid is False
        assert any("INSERT" in e for e in result.errors)

    def test_rejects_drop(self, validator):
        sql = "DROP TABLE tickets"
        result = validator.validate(sql)
        assert result.is_valid is False

    def test_rejects_delete(self, validator):
        sql = "DELETE FROM tickets WHERE 1=1"
        result = validator.validate(sql)
        assert result.is_valid is False

    def test_rejects_no_aggregation(self, validator):
        sql = "SELECT * FROM tickets"
        result = validator.validate(sql)
        assert result.is_valid is False
        assert any("aggregation" in e for e in result.errors)

    def test_rejects_unknown_table(self, validator):
        sql = "SELECT COUNT(*) FROM secret_table GROUP BY id"
        result = validator.validate(sql)
        assert result.is_valid is False
        assert any("secret_table" in e for e in result.errors)

    def test_rejects_multiple_statements(self, validator):
        sql = "SELECT COUNT(*) FROM tickets; DROP TABLE tickets;"
        result = validator.validate(sql)
        assert result.is_valid is False

    def test_valid_top_query(self, validator):
        sql = "SELECT TOP 10 categoryname, COUNT(ticketnumber) AS cnt FROM tickets GROUP BY categoryname ORDER BY cnt DESC"
        result = validator.validate(sql)
        assert result.is_valid is True

    def test_rejects_exec(self, validator):
        sql = "EXEC sp_executesql N'SELECT 1'"
        result = validator.validate(sql)
        assert result.is_valid is False

    def test_valid_join_query(self, validator):
        sql = (
            "SELECT c.categoryname, COUNT(t.ticketnumber) AS cnt "
            "FROM tickets t JOIN categories c ON t.categoryname = c.categoryname "
            "GROUP BY c.categoryname"
        )
        result = validator.validate(sql)
        assert result.is_valid is True
