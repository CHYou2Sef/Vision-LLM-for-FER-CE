# Quick Fix for Cell 3.3

# If you get "ValueError: not enough values to unpack", you need to re-run the Dataset Class cell first.
# Alternatively, use this modified version:

# Select a sample
idx = 0
dataset_item = dataset[idx]

# Handle both 4 and 5 return values
if len(dataset_item) == 5:
    image, prompt, target, label, img_path = dataset_item
else:
    # Fallback if dataset returns only 4 values
    image, prompt, target, label = dataset_item
    # Reconstruct img_path
    img_name = dataset.data.iloc[idx, 0]
    base_name = img_name.split('.')[0]
    img_path = os.path.join(IMG_DIR, f"{base_name}_aligned.jpg")

# Generate explanation
explanation, inputs = infer(image, prompt)

# Generate Grad-CAM
heatmap = get_gradcam_heatmap(model, inputs)

# Overlay visualization
original, heatmap_viz, overlay = overlay_heatmap(img_path, heatmap)

# Analyze coherence
coherence = analyze_explanation_coherence(explanation, heatmap, {})

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(original)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(heatmap_viz)
axes[1].set_title('Grad-CAM Heatmap')
axes[1].axis('off')

axes[2].imshow(overlay)
axes[2].set_title('Overlay')
axes[2].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f'xai_sample_{idx}.png'), dpi=300, bbox_inches='tight')
plt.show()

# Print results
print("="*60)
print("COUCHE 3: XAI INTERPRETATION RESULTS")
print("="*60)
print(f"\nGround Truth: {EMO_MAP[label]}")
print(f"\nGenerated Explanation:\n{explanation}")
print(f"\nMentioned Facial Features: {coherence['mentioned_features']}")
print(f"\nKeywords Found: {coherence['keywords_found']}")
print(f"\nHeatmap Intensity by Region:")
for region, intensity in coherence['heatmap_intensity'].items():
    print(f"  - {region}: {intensity:.4f}")
print("="*60)
