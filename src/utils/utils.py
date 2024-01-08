encoder_key = "encoder.seqTransEncoder"
decoder_key = "decoder.seqTransDecoder"

def load_weights(weight_key:str, pretrain_model_weight, my_model=model):
    # only get the encoder weights
    encoder_weight = {}
    for key in pretrain_model_weight.keys():
        if weight_key in key:
            encoder_weight[key] = pretrain_model_weight[key]

    duplicated_weight = {}
    for phase in PHASES:
        # update weights key name
        for key in encoder_weight.keys():
            if "Encoder" in weight_key:
                name = "encoder"
                new_key = re.sub(r"(encoder)(?=\.)", f"{phase}_{name}", key, count=1)
            elif "Decoder" in weight_key:
                name = "decoder"
                new_key = re.sub(r"(decoder)(?=\.)", f"{phase}_{name}", key, count=1)
                if new_key not in my_model.state_dict().keys():
                    raise Exception(f"Key {new_key} not in model state dict")
            else:
                raise ValueError(f"Weight key {weight_key} not supported")
            duplicated_weight[new_key] = encoder_weight[key]
    # check if the keys are the same
    for key in duplicated_weight.keys():
        if key not in my_model.state_dict().keys():
            print(f"Key {key} not in model state dict")
    # load the state dict to the model
    return duplicated_weight

pretrained_weights = {}
pretrained_weights.update(load_weights(encoder_key, weight))
pretrained_weights.update(load_weights(decoder_key, weight))
model.load_state_dict(pretrained_weights, strict=False)