import nextra from 'nextra'
 
const nextConfig = {}

// Set up Nextra with its configuration
const withNextra = nextra({
  // ... Add Nextra-specific options here
  contentDirBasePath: '/',
})
 
// Export the final Next.js config with Nextra included
export default withNextra(
  // ... Add regular Next.js options here
  nextConfig
)