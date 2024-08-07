AVAILABLE_MODELS = {
    "llava_voco_llama": "VoCoLlamaForCausalLM, VoCoConfig",
}

for model_name, model_classes in AVAILABLE_MODELS.items():
    try:
        exec(f"from .language_model.{model_name} import {model_classes}")
    except ImportError:
        # import traceback
        # traceback.print_exc()
        print(f"Failed to import {model_name} from llava.language_model.{model_name}")
        pass
