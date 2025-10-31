# frozen_string_literal: true

Jekyll::Hooks.register :pages, :pre_render do |page|
  set_default_social_image(page)
end

Jekyll::Hooks.register :posts, :pre_render do |post|
  set_default_social_image(post)
end

Jekyll::Hooks.register :documents, :pre_render do |document|
  set_default_social_image(document)
end

def set_default_social_image(page_or_post)
  return unless page_or_post.data['image'].is_a?(Hash)

  image_data = page_or_post.data['image']

  # If we have light or dark variants but no main image set for social sharing
  if (image_data['light'] || image_data['dark']) && !image_data['path']
    # Prefer light image for social sharing, fallback to dark if light doesn't exist
    social_image = image_data['light'] || image_data['dark']

    # Set the path for jekyll-seo-tag to use
    page_or_post.data['image']['path'] = social_image
  end
end
