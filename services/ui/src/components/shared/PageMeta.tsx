import { Helmet } from 'react-helmet-async';

interface PageMetaProps {
  title: string;
  description: string;
  path: string;
  ogImage?: string;
}

const SITE_NAME = 'SENTINEL';
const DOMAIN = 'https://sentinel.maritime.ai';
const DEFAULT_OG_IMAGE = '/og-image.svg';

export default function PageMeta({ title, description, path, ogImage }: PageMetaProps) {
  const fullTitle = path === '/' ? `${SITE_NAME} — Maritime Oil Spill Intelligence` : `${title} | ${SITE_NAME}`;
  const canonicalUrl = `${DOMAIN}${path}`;
  const ogImageUrl = `${DOMAIN}${ogImage ?? DEFAULT_OG_IMAGE}`;

  return (
    <Helmet>
      <title>{fullTitle}</title>
      <meta name="description" content={description} />
      <link rel="canonical" href={canonicalUrl} />

      {/* Open Graph */}
      <meta property="og:type" content="website" />
      <meta property="og:site_name" content={SITE_NAME} />
      <meta property="og:title" content={fullTitle} />
      <meta property="og:description" content={description} />
      <meta property="og:url" content={canonicalUrl} />
      <meta property="og:image" content={ogImageUrl} />
      <meta property="og:image:width" content="1200" />
      <meta property="og:image:height" content="630" />

      {/* Twitter Card */}
      <meta name="twitter:card" content="summary_large_image" />
      <meta name="twitter:title" content={fullTitle} />
      <meta name="twitter:description" content={description} />
      <meta name="twitter:image" content={ogImageUrl} />
    </Helmet>
  );
}
