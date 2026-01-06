// URL短縮プロキシ Worker
// is.gd APIを使用してURLを短縮

export default {
  async fetch(request, env) {
    // CORS対応
    const corsHeaders = {
      'Access-Control-Allow-Origin': env.ALLOWED_ORIGIN || '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    };

    // OPTIONSリクエスト（プリフライト）
    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders });
    }

    // URLパラメータを取得
    const url = new URL(request.url);
    const longUrl = url.searchParams.get('url');

    if (!longUrl) {
      return new Response(
        JSON.stringify({ error: 'url parameter is required' }),
        {
          status: 400,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' }
        }
      );
    }

    try {
      // is.gd APIを呼び出し
      const apiUrl = `https://is.gd/create.php?format=json&url=${encodeURIComponent(longUrl)}`;
      const response = await fetch(apiUrl);
      const data = await response.json();

      if (data.shorturl) {
        return new Response(
          JSON.stringify({ shortUrl: data.shorturl }),
          {
            headers: { ...corsHeaders, 'Content-Type': 'application/json' }
          }
        );
      } else {
        return new Response(
          JSON.stringify({ error: data.errormessage || 'Failed to shorten URL' }),
          {
            status: 400,
            headers: { ...corsHeaders, 'Content-Type': 'application/json' }
          }
        );
      }
    } catch (e) {
      return new Response(
        JSON.stringify({ error: 'Service unavailable' }),
        {
          status: 500,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' }
        }
      );
    }
  }
};
