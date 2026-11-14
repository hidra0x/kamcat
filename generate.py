import sys
import torch
from datetime import datetime
from diffusers import StableDiffusionPipeline

print("=" * 50)
print("🎨 KamCat Gen v1.0 — генератор картинок")
print("=" * 50)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Устройство: {DEVICE}")

print("Загружаем модель SD 1.5 (при первом запуске скачает ~4 ГБ)...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    safety_checker=None,
)
pipe = pipe.to(DEVICE)

if DEVICE == "cuda":
    pipe.enable_attention_slicing()
    print("✅ Оптимизация VRAM включена")

def generate(prompt, negative_prompt="blurry, low quality, distorted",
             steps=25, guidance=7.5, seed=None):
    if seed is None:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    print(f"\n🎨 Промпт: {prompt}")
    print(f"   Негатив: {negative_prompt}")
    print(f"   Шагов: {steps}, guidance: {guidance}, seed: {seed}")
    print("   Генерация... (10-30 сек)")

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    ).images[0]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"generated_{timestamp}_seed{seed}.png"
    image.save(out_path)
    print(f"   💾 Сохранено: {out_path}")
    return out_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nИспользование: python generate.py \"промпт\"")
        print('Пример: python generate.py "рыжий кот в космосе, цифровое искусство"')
        sys.exit(1)

    prompt = sys.argv[1]
    negative = sys.argv[2] if len(sys.argv) > 2 else "blurry, low quality, distorted"

    generate(prompt, negative)
    print("\n✅ Готово!")