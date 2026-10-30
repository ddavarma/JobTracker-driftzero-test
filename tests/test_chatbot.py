import json
import unittest
from datetime import datetime
from unittest.mock import patch, MagicMock
from openai.types import Model
from openai.types.chat.chat_completion import ChatCompletionMessage
from openai.types.chat.chat_completion_message import FunctionCall
from src.JobTracker.chatbot import ChatBot, ChatGPT

MODEL_LIST_RETURN = [
    Model(
        id="gpt-3.5-turbo",
        created=1677610602,
        object="model",
        owned_by="openai",
    ),
    Model(
        id="gpt-4-1106-preview",
        created=1691417244,
        object="model",
        owned_by="openai",
    ),
]

class TestChatBot(unittest.TestCase):

    @patch('src.JobTracker.chatbot.OpenAI')
    def test_gen_prompt(self, mock_openai_client):
        mock_openai_client.return_value.models.list.return_value = MODEL_LIST_RETURN
        chat_bot = ChatGPT()
        info = "This is a test email body."
        expected_threshold = 80  # Replace with the actual threshold value used in your method
        expected_output = (
            f'If this email has a {expected_threshold}% confidence level that it is from a company I applied to or interviewed with before, '
            f'use get_mail_info to get information, otherwise just ignore it. Here is the mail body: {info}')
        self.assertEqual(chat_bot.gen_prompt(info), expected_output)


class TestChatGPT(unittest.TestCase):

    @patch('src.JobTracker.chatbot.OpenAI')
    def setUp(self, mock_openai_client):
        mock_openai_client.return_value.models.list.return_value = MODEL_LIST_RETURN
        self.chat_gpt = ChatGPT()

    @patch('src.JobTracker.config')
    def test_get_content_succeed(self, mock_config):
        info = {'body': 'test email body', 'date': "Thu, 09 Nov 2023 23:27:06 +0000"}
        mock_config.API_KEY = 'test_api_key'
        mock_config.FUNCTION = 'test_function'
        # Mock the API response
        mock_choice = MagicMock()
        mock_choice.finish_reason = 'function_call'
        mock_choice.message = ChatCompletionMessage(
            role='assistant',
            content=None,
            function_call=FunctionCall(
                name='get_email_info',
                arguments=json.dumps({
                    'company': 'TestCompany',
                    'state': 'TestState',
                    'next_step': 'TestNextStep'
                }),
            ),
        )

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        self.chat_gpt.client.chat.completions.create.return_value = mock_response
        state, data = self.chat_gpt.get_content(info)
        date_object = datetime.strptime(info['date'], "%a, %d %b %Y %H:%M:%S %z")
        month_day_year_time = date_object.strftime("%b %d %Y %H:%M:%S")
        self.assertEqual(state, 'Succeed')
        self.assertEqual(data['company'], 'TestCompany')
        self.assertEqual(data['state'], json.dumps({"TestState": month_day_year_time}))
        self.assertEqual(data['next_step'], 'TestNextStep')

    @patch('src.JobTracker.config')
    def test_get_content_failed(self, mock_config):
        info = {'body': 'test email body'}
        mock_config.API_KEY = 'test_api_key'
        mock_config.FUNCTION = 'test_function'
        # Mock the API response
        mock_choice = MagicMock()
        mock_choice.finish_reason = 'other'
        mock_choice.message = ChatCompletionMessage(role='assistant', content=None)
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        self.chat_gpt.client.chat.completions.create.return_value = mock_response

        state, data = self.chat_gpt.get_content(info)

        self.assertEqual(state, 'Failed')
        self.assertEqual(data, 'Not related to a job application or interview process')


if __name__ == '__main__':
    unittest.main()
