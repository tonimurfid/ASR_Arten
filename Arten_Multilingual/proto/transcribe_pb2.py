# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: proto/transcribe.proto
# Protobuf Python Version: 5.27.2
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    27,
    2,
    '',
    'proto/transcribe.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x16proto/transcribe.proto\x12\rtranscription\")\n\x14TranscriptionRequest\x12\x11\n\tfile_data\x18\x01 \x01(\x0c\"6\n\x15TranscriptionResponse\x12\x1d\n\x15transcription_message\x18\x01 \x01(\t2{\n\x1b\x45nglishTranscriptionService\x12\\\n\x0fTranscribeAudio\x12#.transcription.TranscriptionRequest\x1a$.transcription.TranscriptionResponse2~\n\x1eIndonesianTranscriptionService\x12\\\n\x0fTranscribeAudio\x12#.transcription.TranscriptionRequest\x1a$.transcription.TranscriptionResponseB\x15Z\x13model/transcriptionb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'proto.transcribe_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  _globals['DESCRIPTOR']._loaded_options = None
  _globals['DESCRIPTOR']._serialized_options = b'Z\023model/transcription'
  _globals['_TRANSCRIPTIONREQUEST']._serialized_start=41
  _globals['_TRANSCRIPTIONREQUEST']._serialized_end=82
  _globals['_TRANSCRIPTIONRESPONSE']._serialized_start=84
  _globals['_TRANSCRIPTIONRESPONSE']._serialized_end=138
  _globals['_ENGLISHTRANSCRIPTIONSERVICE']._serialized_start=140
  _globals['_ENGLISHTRANSCRIPTIONSERVICE']._serialized_end=263
  _globals['_INDONESIANTRANSCRIPTIONSERVICE']._serialized_start=265
  _globals['_INDONESIANTRANSCRIPTIONSERVICE']._serialized_end=391
# @@protoc_insertion_point(module_scope)