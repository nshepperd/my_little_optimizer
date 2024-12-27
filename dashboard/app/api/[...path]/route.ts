// app/api/[...path]/route.ts

// Handle all HTTP methods
const handler = async (req: Request) => {
    const url = new URL(req.url);
    const path = url.pathname.replace('/api/', '');
    
    const response = await fetch(`http://localhost:8000/api/${path}`, {
      method: req.method,
      headers: {
        'Content-Type': 'application/json',
        // Forward other relevant headers if needed
      },
      ...(req.method !== 'GET' && req.method !== 'HEAD' ? { body: await req.text() } : {}),
    });
  
    return response;
  };
  
  // Export a handler for each HTTP method
  export const GET = handler;
  export const POST = handler;
  export const PUT = handler;
  export const DELETE = handler;
  export const PATCH = handler;
  export const HEAD = handler;
  export const OPTIONS = handler;