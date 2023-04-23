from fitparse import FitFile
fitfile = FitFile('data/20201220013535730_45696.fit')
# Get all data messages that are of type record
for record in fitfile.get_messages('record'):
    # Go through all the data entries in this record
    for record_data in record:

        # Print the records name and value (and units if it has any)
        if record_data.units:
            print(" * %s: %s %s" % (
                record_data.name, record_data.value, record_data.units,
            ))
        else:
            print(" * %s: %s" % (record_data.name, record_data.value))


# import fitdecode
#
# with fitdecode.FitReader('data/20201220013535730_45696.fit') as fit:
#     print(fit)
#
#     for frame in fit:
#         # The yielded frame object is of one of the following types:
#         # * fitdecode.FitHeader (FIT_FRAME_HEADER)
#         # * fitdecode.FitDefinitionMessage (FIT_FRAME_DEFINITION)
#         # * fitdecode.FitDataMessage (FIT_FRAME_DATA)
#         # * fitdecode.FitCRC (FIT_FRAME_CRC)
#
#         if frame.frame_type == fitdecode.FIT_FRAME_DATA:
#             # Here, frame is a FitDataMessage object.
#             # A FitDataMessage object contains decoded values that
#             # are directly usable in your script logic.
#             print(frame.name)