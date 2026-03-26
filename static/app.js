const searchInput = document.getElementById("cliente-busca");
const hiddenCode = document.getElementById("codigo-cliente");
const list = document.getElementById("autocomplete-list");
const clientCard = document.getElementById("cliente-card");
const clientName = document.getElementById("cliente-nome");
const clientMeta = document.getElementById("cliente-meta");
const clientCanal = document.getElementById("cliente-canal");

if (searchInput && hiddenCode && list) {
    const hideList = () => {
        list.classList.add("hidden");
        list.innerHTML = "";
    };

    const fillClient = (client) => {
        searchInput.value = `${client.nome} (${client.codigo})`;
        hiddenCode.value = client.codigo;
        clientName.textContent = client.nome;
        clientMeta.textContent = `${client.cnpj} | ${client.cidade}/${client.uf} | ${client.endereco}`;
        clientCanal.textContent = client.canal || "Sem canal";
        clientCard.classList.remove("hidden");
        hideList();
    };

    searchInput.addEventListener("input", async (event) => {
        const query = event.target.value.trim();
        hiddenCode.value = "";
        clientCard.classList.add("hidden");
        if (query.length < 3) {
            hideList();
            return;
        }

        const response = await fetch(`/api/clientes?q=${encodeURIComponent(query)}`);
        if (!response.ok) {
            hideList();
            return;
        }

        const items = await response.json();
        if (!items.length) {
            list.innerHTML = '<div class="autocomplete-item"><strong>Nenhum cliente encontrado</strong><span>Continue digitando ou revise o codigo/CNPJ.</span></div>';
            list.classList.remove("hidden");
            return;
        }

        list.innerHTML = "";
        items.forEach((client) => {
            const button = document.createElement("button");
            button.type = "button";
            button.className = "autocomplete-item";
            button.innerHTML = `<strong>${client.nome}</strong><span>${client.codigo} | ${client.cnpj} | ${client.cidade}/${client.uf}</span>`;
            button.addEventListener("click", () => fillClient(client));
            list.appendChild(button);
        });
        list.classList.remove("hidden");
    });

    document.addEventListener("click", (event) => {
        if (!event.target.closest(".autocomplete-field")) {
            hideList();
        }
    });
}
