import StructuredData from './StructuredData';

const WEB_APPLICATION_SCHEMA = {
  '@context': 'https://schema.org',
  '@type': 'WebApplication',
  name: 'SENTINEL',
  description: 'Autonomous Maritime Oil Spill Intelligence Platform. Detect, attribute, and investigate maritime oil spills using SAR imagery, AIS tracking, and AI analysis.',
  url: 'https://sentinel.maritime.ai',
  applicationCategory: 'SecurityApplication',
  operatingSystem: 'Web',
  offers: {
    '@type': 'Offer',
    price: '0',
    priceCurrency: 'USD',
  },
  creator: {
    '@type': 'Organization',
    name: 'SENTINEL Maritime Intelligence',
    url: 'https://sentinel.maritime.ai',
  },
};

const ORGANIZATION_SCHEMA = {
  '@context': 'https://schema.org',
  '@type': 'Organization',
  name: 'SENTINEL Maritime Intelligence',
  url: 'https://sentinel.maritime.ai',
  logo: 'https://sentinel.maritime.ai/favicon.svg',
  description: 'Maritime oil spill detection, attribution, and investigation using satellite imagery and artificial intelligence.',
  contactPoint: {
    '@type': 'ContactPoint',
    contactType: 'technical support',
    availableLanguage: 'English',
  },
  sameAs: [],
};

const LOCAL_BUSINESS_SCHEMA = {
  '@context': 'https://schema.org',
  '@type': 'LocalBusiness',
  name: 'SENTINEL Maritime Intelligence',
  description: 'Maritime oil spill detection and investigation platform providing autonomous surveillance and intelligence services.',
  url: 'https://sentinel.maritime.ai',
  image: 'https://sentinel.maritime.ai/og-image.svg',
  address: {
    '@type': 'PostalAddress',
    addressCountry: 'International',
  },
  geo: {
    '@type': 'GeoCoordinates',
    latitude: 15.0,
    longitude: 73.0,
  },
  openingHoursSpecification: {
    '@type': 'OpeningHoursSpecification',
    dayOfWeek: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
    opens: '00:00',
    closes: '23:59',
  },
};

export default function SiteSchema() {
  return (
    <>
      <StructuredData data={WEB_APPLICATION_SCHEMA} />
      <StructuredData data={ORGANIZATION_SCHEMA} />
      <StructuredData data={LOCAL_BUSINESS_SCHEMA} />
    </>
  );
}
