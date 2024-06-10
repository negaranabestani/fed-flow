from unittest import TestCase

from app.dto.message import GlobalWeightMessage, BaseMessage, MessageType, JsonMessage
from app.model.entity.nn_model.vgg import VGG


class MessageTest(TestCase):
    def test_create_message_by_type_should_return_appropriate_message_type_when_type_exists(self):
        # Given
        msg_type = GlobalWeightMessage.MESSAGE_TYPE

        # When
        msg = BaseMessage.create_message_by_type(msg_type, None)

        # Then
        self.assertIsInstance(msg, GlobalWeightMessage)

    def test_create_message_by_type_should_raise_error_when_given_type_not_exist(self):
        # Given
        msg_type = MessageType('INVALID_MESSAGE_TYPE')

        # When, Then
        with self.assertRaises(ValueError):
            BaseMessage.create_message_by_type(msg_type, None)

    def test_deserialize_global_weight_should_return_same_object(self):
        # Given
        weights = [VGG().initialize('', [1], True).state_dict()]
        msg = GlobalWeightMessage(weights)

        # When
        serialized_msg = msg.serialize()
        deserialized_msg = BaseMessage.deserialize(serialized_msg, GlobalWeightMessage.MESSAGE_TYPE)

        # Then
        self.assertIsInstance(deserialized_msg, GlobalWeightMessage)
        self.assertEqual(len(deserialized_msg.weights), len(weights))

    def test_deserialize_json_message_should_return_same_object(self):
        # Given
        data = {'key': 'value'}
        msg = JsonMessage(data)

        # When
        serialized_msg = msg.serialize()
        deserialized_msg = BaseMessage.deserialize(serialized_msg, JsonMessage.MESSAGE_TYPE)

        # Then
        self.assertDictEqual(data, deserialized_msg.data)
