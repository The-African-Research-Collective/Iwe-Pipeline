from contextlib import nullcontext

from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.utils.typeshelper import StatHints


class AzureContentMD5DedupFilter(PipelineStep):
    type = "🫂 - DEDUPS"
    name = "💥 content-md5-deduplication"

    def __init__(
        self,
        exclusion_writer: DiskWriter | None = None,
    ):
        super().__init__()
        self.exclusion_writer = exclusion_writer

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        """
        content_md5 is in data.metadata["source"] dict

        it has been base64 encoded as follows before being placed in the dict:
        content_md5_str = (
            base64.b64encode(content_md5_bytes).decode("utf-8") if content_md5_bytes else None
        )
        """
        seen = set()

        with self.exclusion_writer if self.exclusion_writer else nullcontext():
            for doc in data:
                self.stat_update(StatHints.total)

                content_md5 = doc.metadata.get("source", {}).get("content_md5")

                if not content_md5:
                    # No hash available — let it through
                    self.stat_update(StatHints.forwarded)
                    yield doc
                    continue

                if content_md5 in seen:
                    self.stat_update(StatHints.dropped)
                    if self.exclusion_writer:
                        self.exclusion_writer.write(doc, rank)
                    continue

                seen.add(content_md5)
                self.stat_update(StatHints.forwarded)
                self.update_doc_stats(doc)
                yield doc
